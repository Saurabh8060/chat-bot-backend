import os
import re
import difflib
import logging
from dataclasses import dataclass
from typing import List, Tuple

from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import NotFoundException
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch

from .prompt import RAG_PROMPT

logger = logging.getLogger("rag-helper")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "gcp")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-central1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-chatbot")

# NOTE: LLM settings are read at call time to respect .env reloads.

TOP_K = int(os.getenv("TOP_K", "4"))
PDF_DIR = os.getenv("PDF_DIR", "./pdfs")

_pc = None
_vectorstore = None
_index_ready = False
_hf_model = None
_hf_tokenizer = None
_hf_is_encoder_decoder = None


@dataclass
class _LLMResponse:
    content: str


def download_embeddings() -> HuggingFaceEmbeddings:
    """Download and return the HuggingFace embeddings model."""
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings


def _get_embeddings() -> HuggingFaceEmbeddings:
    return download_embeddings()


def ensure_vectorstore() -> None:
    global _pc, _vectorstore, _index_ready
    if _vectorstore is not None:
        return

    embeddings = _get_embeddings()
    embedding_dim = int(os.getenv("EMBEDDING_DIM", "768"))

    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is not set. FAISS fallback is disabled.")

    if _pc is None:
        _pc = Pinecone(api_key=PINECONE_API_KEY)
    indexes = [idx["name"] for idx in _pc.list_indexes()]
    if PINECONE_INDEX_NAME not in indexes:
        _pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=embedding_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    else:
        try:
            info = _pc.describe_index(PINECONE_INDEX_NAME)
            if getattr(info, "dimension", None) not in (None, embedding_dim):
                raise RuntimeError(
                    f"Pinecone index '{PINECONE_INDEX_NAME}' has dimension {info.dimension}, "
                    f"but embeddings are {embedding_dim}. Create a new index or change PINECONE_INDEX_NAME."
                )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to validate Pinecone index dimensions: {exc}"
            ) from exc
    _vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        text_key="text",
        namespace="default",
    )
    _index_ready = True


def split_text(text: str) -> List[str]:
    cleaned = _clean_text(text)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = [c.strip() for c in splitter.split_text(cleaned) if c.strip()]
    return _filter_chunks(chunks)


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r", "\n")
    # Fix hyphenated line breaks: "develop-\nment" -> "development"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    lines = [ln.strip() for ln in text.split("\n")]
    cleaned_lines = []
    for ln in lines:
        if not ln:
            cleaned_lines.append("")
            continue
        if _is_boilerplate_line(ln):
            continue
        cleaned_lines.append(ln)

    # Collapse multiple empty lines
    cleaned = []
    prev_empty = False
    for ln in cleaned_lines:
        is_empty = not ln
        if is_empty and prev_empty:
            continue
        cleaned.append(ln)
        prev_empty = is_empty

    text = "\n".join(cleaned)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _clean_answer_text(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    # Remove leading bullets/stray punctuation from dataset artifacts
    text = re.sub(r"^[\)\]\}\.\,\-–•\s]+", "", text)
    return text.strip()


def _is_boilerplate_line(line: str) -> bool:
    line_l = line.lower()
    if re.fullmatch(r"\d{1,4}", line_l):
        return True
    if line_l.startswith("page ") and re.search(r"\b\d+\b", line_l):
        return True
    if re.search(r"gem\s*-\d+\s*to\s*\d+.*page\s*\d+", line_l):
        return True
    if line_l.startswith("copyright") or "all rights reserved" in line_l:
        return True
    if len(line_l) < 3:
        return True
    # Skip lines with very low alphabetic content
    letters = sum(ch.isalpha() for ch in line)
    if letters / max(len(line), 1) < 0.3 and len(line) < 80:
        return True
    return False


def _filter_chunks(chunks: List[str]) -> List[str]:
    seen = set()
    filtered: List[str] = []
    for chunk in chunks:
        norm = re.sub(r"\s+", " ", chunk.lower()).strip()
        if not norm or norm in seen:
            continue
        seen.add(norm)

        words = chunk.split()
        if len(words) < 30:
            continue
        letters = sum(ch.isalpha() for ch in chunk)
        if letters / max(len(chunk), 1) < 0.5:
            continue
        filtered.append(chunk)
    return filtered


def ingest_text(chunks: List[str], replace: bool = False) -> int:
    ensure_vectorstore()
    if replace:
        reset_store()

    docs = [Document(page_content=chunk, metadata={"source": "pdf"}) for chunk in chunks]
    _vectorstore.add_documents(docs)  # type: ignore[union-attr]
    return len(docs)


def ingest_qa(pairs: List[tuple[str, str]], replace: bool = False) -> int:
    ensure_vectorstore()
    if replace:
        reset_store()

    docs = []
    for q, a in pairs:
        q = q.strip()
        a = a.strip()
        if not q or not a:
            continue
        docs.append(Document(page_content=q, metadata={"answer": a, "source": "medquad"}))

    if not docs:
        return 0
    _vectorstore.add_documents(docs)  # type: ignore[union-attr]
    return len(docs)


def reset_store() -> None:
    global _vectorstore
    if not _index_ready or _vectorstore is None:
        return
    try:
        _vectorstore.delete(namespace="default", delete_all=True)
    except NotFoundException:
        # Namespace doesn't exist yet; nothing to delete.
        logger.info("Pinecone namespace 'default' not found; skipping delete.")


def get_retrieved_docs(question: str) -> List[Document]:
    ensure_vectorstore()
    retriever = _vectorstore.as_retriever(search_kwargs={"k": TOP_K})  # type: ignore[union-attr]
    return retriever.invoke(question)


def _get_llm():
    llm_provider = os.getenv("LLM_PROVIDER", "").lower().strip()
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
    hf_model = os.getenv("HF_MODEL", "google/flan-t5-base")
    hf_max_new_tokens = int(os.getenv("HF_MAX_NEW_TOKENS", "512"))
    hf_temperature = float(os.getenv("HF_TEMPERATURE", "0"))
    hf_top_p = float(os.getenv("HF_TOP_P", "0.95"))
    hf_device = int(os.getenv("HF_DEVICE", "-1"))

    if llm_provider == "hf":
        logger.info("LLM_PROVIDER=hf -> using Hugging Face model %s", hf_model)
        return _get_hf_llm(
            model_name=hf_model,
            max_new_tokens=hf_max_new_tokens,
            temperature=hf_temperature,
            top_p=hf_top_p,
            device=hf_device,
        )

    # Prefer OpenAI when an API key is present unless explicitly forced to Ollama.
    if openai_api_key and llm_provider not in ("ollama", "hf"):
        logger.info("LLM_PROVIDER=%s -> using OpenAI", llm_provider or "auto")
        return ChatOpenAI(model=openai_model, temperature=0, openai_api_key=openai_api_key)

    logger.info("LLM_PROVIDER=%s -> using Ollama", llm_provider or "ollama")
    if llm_provider == "ollama" or not llm_provider:
        return ChatOllama(model=ollama_model, temperature=0)
    return None


def _get_hf_llm(model_name: str, max_new_tokens: int, temperature: float, top_p: float, device: int):
    global _hf_model, _hf_tokenizer, _hf_is_encoder_decoder
    if _hf_model is None or _hf_tokenizer is None or _hf_is_encoder_decoder is None:
        _hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = None
        try:
            config = AutoModelForSeq2SeqLM.from_pretrained(model_name).config
            _hf_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            _hf_is_encoder_decoder = True
        except Exception:
            _hf_model = AutoModelForCausalLM.from_pretrained(model_name)
            _hf_is_encoder_decoder = False

        if device is not None and device >= 0 and torch.cuda.is_available():
            _hf_model = _hf_model.to(f"cuda:{device}")
        else:
            _hf_model = _hf_model.to("cpu")

    class _HFWrapper:
        def __init__(self):
            self.max_new_tokens = max_new_tokens
            self.temperature = temperature
            self.top_p = top_p

        def invoke(self, prompt: str):
            do_sample = self.temperature > 0
            inputs = _hf_tokenizer(prompt, return_tensors="pt", truncation=True)
            inputs = {k: v.to(_hf_model.device) for k, v in inputs.items()}
            gen_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": do_sample,
            }
            if do_sample:
                gen_kwargs["temperature"] = self.temperature
                gen_kwargs["top_p"] = self.top_p

            with torch.no_grad():
                outputs = _hf_model.generate(**inputs, **gen_kwargs)
            text = _hf_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            # For causal models, strip echoed prompt.
            if not _hf_is_encoder_decoder:
                if text.startswith(prompt):
                    text = text[len(prompt):].strip()
                if "Answer:" in text:
                    text = text.split("Answer:", 1)[-1].strip()
            return _LLMResponse(content=text)

    return _HFWrapper()


def answer_question(question: str, last_topic: str | None = None) -> dict:
    if _is_greeting(question):
        return {"answer": "Please ask a medical question.", "sources": []}

    docs, best_score = _retrieve_with_scores(question)
    if not docs:
        return {"answer": "I don't know. The answer isn't in the provided context.", "sources": []}

    if _needs_clarification(question):
        if last_topic:
            question = f"{question} (about: {last_topic})"
        else:
            return {"answer": "Please specify the condition or topic you're asking about.", "sources": []}

    # Refresh retrieval if we expanded the question with context.
    docs, best_score = _retrieve_with_scores(question)
    if not docs:
        return {"answer": "I don't know. The answer isn't in the provided context.", "sources": []}

    min_score = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.15"))
    if best_score is not None and best_score < min_score:
        return {"answer": "I don't know. The answer isn't in the provided context.", "sources": []}

    # Re-rank docs by fuzzy match against the Q line (if present).
    docs = _rerank_docs_by_question(question, docs)
    min_q_match = float(os.getenv("MIN_QUESTION_MATCH", "0.42"))
    if docs and docs[0].metadata.get("_qmatch", 0.0) < min_q_match:
        return {"answer": "I don't know. The answer isn't in the provided context.", "sources": []}

    # Build context from top documents (Q/A pairs for MedQuAD).
    context = _top_k_docs_text(docs)
    if not context:
        return {"answer": "I don't know. The answer isn't in the provided context.", "sources": []}
    llm = _get_llm()
    if llm is None:
        return {"answer": "I don't know. The answer isn't in the provided context.", "sources": []}

    prompt = RAG_PROMPT.format(context=context, question=question)
    response = llm.invoke(prompt)
    answer = _sanitize_llm_output(response.content, prompt, context)
    answer = _clean_answer_text(answer)
    if not answer:
        return {"answer": "I don't know. The answer isn't in the provided context.", "sources": []}

    if _sentence_count(answer) < 2:
        expand_prompt = (
            "Rewrite the answer in 2–4 sentences using ONLY the context. "
            "Do not add new facts.\n\n"
            f"Context:\n{context}\n\n"
            f"Answer:\n{answer}\n\n"
            "Expanded answer:"
        )
        expanded = llm.invoke(expand_prompt)
        expanded_text = _sanitize_llm_output(expanded.content, expand_prompt, context)
        if expanded_text and _sentence_count(expanded_text) >= 2:
            answer = _clean_answer_text(expanded_text)
        else:
            extra = _second_sentence_from_context(answer, context)
            if extra:
                answer = f"{answer.rstrip('. ')}. {extra}"

    return {
        "answer": answer,
        "sources": [{"text": d.page_content, "source": d.metadata.get("source", "")} for d in docs],
    }


def _top_k_docs_text(docs: List[Document]) -> str:
    k = int(os.getenv("TOP_K_DOCS", "4"))
    parts = []
    for d in docs[:k]:
        if d.page_content:
            answer = (d.metadata or {}).get("answer") if hasattr(d, "metadata") else None
            if isinstance(answer, str) and answer.strip():
                parts.append(_clean_answer_text(answer))
            else:
                extracted = _extract_doc_answer(d.page_content)
                if extracted:
                    parts.append(extracted)
                else:
                    parts.append(d.page_content.strip())
    return "\n\n".join(parts).strip()


def _rerank_docs_by_question(question: str, docs: List[Document]) -> List[Document]:
    q_norm = _normalize_text(question)
    ranked = []
    for d in docs:
        q_line = _extract_doc_question(d.page_content)
        target = _normalize_text(q_line or d.page_content)
        score = difflib.SequenceMatcher(None, q_norm, target).ratio()
        d.metadata = dict(d.metadata or {})
        d.metadata["_qmatch"] = score
        ranked.append((score, d))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in ranked]


def _extract_doc_question(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("q:"):
            return line[2:].strip()
    return ""


def _extract_doc_answer(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    collecting = False
    answer_parts = []
    for ln in lines:
        if ln.lower().startswith("a:"):
            collecting = True
            answer_parts.append(ln[2:].strip())
            continue
        if ln.lower().startswith("q:"):
            if collecting:
                break
            continue
        if collecting:
            answer_parts.append(ln)
    answer = " ".join(answer_parts).strip()
    return answer


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _retrieve_with_scores(question: str) -> Tuple[List[Document], float | None]:
    ensure_vectorstore()
    # Try to use scores if supported by the vectorstore.
    try:
        results = _vectorstore.similarity_search_with_score(question, k=TOP_K)  # type: ignore[union-attr]
        if not results:
            return [], None
        docs = [d for d, _ in results]
        # Pinecone returns similarity; higher is better.
        best_score = max(score for _, score in results)
        return docs, float(best_score)
    except Exception:
        docs = get_retrieved_docs(question)
        return docs, None




def _is_greeting(text: str) -> bool:
    t = text.strip().lower()
    return t in {"hi", "hello", "hey", "yo"}


def _needs_clarification(question: str) -> bool:
    q = question.lower().strip()
    tokens = [t for t in re.split(r"\W+", q) if t]
    if len(tokens) < 3:
        return True
    if "talking about" in q or "we talking" in q or "discussing" in q:
        return True
    if tokens and tokens[-1] in {"it", "this", "that", "they"}:
        return True
    if any(t in {"it", "this", "that"} for t in tokens) and len(tokens) <= 4:
        return True
    return False


def _sanitize_llm_output(text: str, prompt: str, context: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    # Remove full prompt if echoed.
    if cleaned.startswith(prompt):
        cleaned = cleaned[len(prompt):].strip()
    # Remove raw context if echoed.
    if cleaned.startswith(context):
        cleaned = cleaned[len(context):].strip()
    if "Answer:" in cleaned:
        cleaned = cleaned.split("Answer:", 1)[-1].strip()
    # Final cleanup of repeated whitespace.
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _sentence_count(text: str) -> int:
    parts = [s for s in re.split(r"[.!?]+\s+", text.strip()) if s]
    return len(parts)


def _second_sentence_from_context(answer: str, context: str) -> str:
    answer_tokens = _content_tokens(answer)
    for sent in re.split(r"(?<=[.!?])\s+", context):
        sent = sent.strip()
        if len(sent) < 20:
            continue
        s_tokens = _content_tokens(sent)
        if not s_tokens:
            continue
        # pick a sentence that adds new info
        if len(s_tokens - answer_tokens) >= 3:
            return sent.rstrip(". ")
    return ""


def _content_tokens(text: str) -> set[str]:
    stop = {
        "what", "was", "about", "talking", "talk", "we", "you", "i", "me", "my",
        "the", "a", "an", "is", "are", "was", "were", "this", "that", "it",
        "how", "why", "when", "where", "which", "who", "whom", "do", "does",
        "did", "can", "could", "should", "would", "please"
    }
    return {t for t in re.split(r"\W+", text.lower()) if len(t) > 2 and t not in stop}
