import os
import re
import logging
from dataclasses import dataclass
from typing import List, Tuple

from pinecone import Pinecone, ServerlessSpec
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
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings


def _get_embeddings() -> HuggingFaceEmbeddings:
    return download_embeddings()


def ensure_vectorstore() -> None:
    global _pc, _vectorstore, _index_ready
    if _vectorstore is not None:
        return

    embeddings = _get_embeddings()

    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is not set. FAISS fallback is disabled.")

    if _pc is None:
        _pc = Pinecone(api_key=PINECONE_API_KEY)
    indexes = [idx["name"] for idx in _pc.list_indexes()]
    if PINECONE_INDEX_NAME not in indexes:
        _pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    else:
        try:
            info = _pc.describe_index(PINECONE_INDEX_NAME)
            if getattr(info, "dimension", None) not in (None, 384):
                raise RuntimeError(
                    f"Pinecone index '{PINECONE_INDEX_NAME}' has dimension {info.dimension}, "
                    "but embeddings are 384. Create a new index or change PINECONE_INDEX_NAME."
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


def reset_store() -> None:
    global _vectorstore
    if not _index_ready or _vectorstore is None:
        return
    _vectorstore.delete(namespace="default", delete_all=True)


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


def answer_question(question: str) -> dict:
    docs = get_retrieved_docs(question)
    if not docs:
        return {"answer": "I don't know. The answer isn't in the provided context.", "sources": []}

    # Build a short, high-signal context: top-K sentences from retrieved docs.
    context = _top_k_sentences(question, docs)
    if not context:
        return {"answer": "I don't know. The answer isn't in the provided context.", "sources": []}
    llm = _get_llm()
    if llm is None:
        return {"answer": "I don't know. The answer isn't in the provided context.", "sources": []}

    prompt = RAG_PROMPT.format(context=context, question=question)
    response = llm.invoke(prompt)
    answer = _sanitize_llm_output(response.content, prompt, context)
    if not answer:
        return {"answer": "I don't know. The answer isn't in the provided context.", "sources": []}

    return {
        "answer": answer,
        "sources": [{"text": d.page_content, "source": d.metadata.get("source", "")} for d in docs],
    }


def _top_k_sentences(question: str, docs: List[Document]) -> str:
    k = int(os.getenv("TOP_K_SENTENCES", "10"))
    q_tokens = {t for t in re.split(r"\W+", question.lower()) if len(t) > 2}
    sentences: List[Tuple[str, int]] = []
    for d in docs:
        for sent in re.split(r"(?<=[.!?])\s+", d.page_content):
            sent = sent.strip()
            if len(sent) < 20:
                continue
            s_tokens = {t for t in re.split(r"\W+", sent.lower()) if len(t) > 2}
            score = len(q_tokens & s_tokens)
            sentences.append((sent, score))

    if not sentences:
        return ""

    # Sort by score then length, keep top-k.
    sentences.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
    top = [s for s, _ in sentences[:k]]

    # If all scores are zero, fall back to the first k sentences from docs.
    if sentences[0][1] == 0:
        fallback = []
        for d in docs:
            for sent in re.split(r"(?<=[.!?])\s+", d.page_content):
                sent = sent.strip()
                if len(sent) < 20:
                    continue
                fallback.append(sent)
                if len(fallback) >= k:
                    break
            if len(fallback) >= k:
                break
        top = fallback or top

    return " ".join(top).strip()


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
