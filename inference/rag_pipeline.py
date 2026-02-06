import os
import logging
import time
import concurrent.futures
import re
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# Config (override with env vars)
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
BASE_MODEL = os.getenv("BASE_MODEL", "google/flan-t5-small")
LORA_PATH = os.getenv("LORA_PATH", "lora_model")

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss.index")
CHUNKS_PATH = os.getenv("CHUNKS_PATH", "chunks.npy")
META_PATH = os.getenv("META_PATH", "chunks_meta.json")

TOP_K = int(os.getenv("TOP_K", "2"))
FAST_MODE = os.getenv("FAST_MODE", "false").lower() == "true"
FORCE_EXTRACTIVE = os.getenv("FORCE_EXTRACTIVE", "true").lower() == "true"
LOCAL_ONLY = os.getenv("LOCAL_ONLY", "false").lower() == "true"
FALLBACK_EMBEDDINGS = os.getenv("FALLBACK_EMBEDDINGS", "true").lower() == "true"
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "60"))
CONTEXT_CHARS = int(os.getenv("CONTEXT_CHARS", "800"))
GEN_MAX_NEW_TOKENS = int(os.getenv("GEN_MAX_NEW_TOKENS", "40"))
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.05"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger = logging.getLogger("rag-pipeline")

try:
    embedder = SentenceTransformer(EMBED_MODEL, local_files_only=LOCAL_ONLY)
except Exception:
    logger.warning("Embedding model not available locally (LOCAL_ONLY=%s)", LOCAL_ONLY)
    embedder = None

FALLBACK_DIM = 256


def _summarize_extractively(text: str, max_sentences: int = 3) -> str:
    if not text:
        return ""
    # Basic sentence split
    raw_sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    if not raw_sentences:
        return ""
    if len(raw_sentences) <= max_sentences:
        summary = ". ".join(raw_sentences)
        return summary + "." if summary and not summary.endswith(".") else summary

    # Score sentences by word frequency
    stopwords = {
        "the","is","are","a","an","and","or","of","to","in","for","on","with","as","by",
        "at","from","this","that","it","be","was","were","has","have","had","but","not",
        "you","your","we","they","their","our","its","if","then","so","than","into","about"
    }
    word_freq: dict[str, int] = {}
    for sentence in raw_sentences:
        for token in sentence.lower().split():
            token = "".join(ch for ch in token if ch.isalnum())
            if not token or token in stopwords:
                continue
            word_freq[token] = word_freq.get(token, 0) + 1

    scored = []
    for idx, sentence in enumerate(raw_sentences):
        score = 0
        for token in sentence.lower().split():
            token = "".join(ch for ch in token if ch.isalnum())
            if not token:
                continue
            score += word_freq.get(token, 0)
        scored.append((score, idx, sentence))

    scored.sort(reverse=True, key=lambda x: x[0])
    top = sorted(scored[:max_sentences], key=lambda x: x[1])
    summary = ". ".join(s for _, _, s in top)
    return summary + "." if summary and not summary.endswith(".") else summary


def _extractive_answer(question: str, context: str) -> str:
    if not context:
        return ""
    question_tokens = {t for t in re.split(r"\W+", question.lower()) if len(t) > 2}
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if s.strip()]
    best = ("", 0)
    for s in sentences:
        tokens = {t for t in re.split(r"\W+", s.lower()) if len(t) > 2}
        score = len(tokens & question_tokens)
        if score > best[1]:
            best = (s, score)
    if best[0]:
        return best[0]
    return _summarize_extractively(context)


def _fallback_embed(texts: list[str]) -> np.ndarray:
    vectors = np.zeros((len(texts), FALLBACK_DIM), dtype="float32")
    for i, text in enumerate(texts):
        for token in text.lower().split():
            idx = hash(token) % FALLBACK_DIM
            vectors[i, idx] += 1.0
    # normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms

index = None
chunks: list[str] = []
sources: list[str] = []


def load_store() -> None:
    global index, chunks, sources
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        chunks = np.load(CHUNKS_PATH, allow_pickle=True).tolist()
        if os.path.exists(META_PATH):
            import json
            with open(META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
            sources = meta.get("sources", [])
        else:
            sources = [""] * len(chunks)
        logger.info("Loaded index with %d chunks", len(chunks))
    else:
        index = None
        chunks = []
        sources = []
        logger.warning("No index found yet. You must ingest documents.")


def save_store() -> None:
    if index is None:
        return
    faiss.write_index(index, FAISS_INDEX_PATH)
    np.save(CHUNKS_PATH, np.array(chunks, dtype=object))
    import json
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"sources": sources}, f)


def reset_store() -> None:
    global index, chunks, sources
    index = None
    chunks = []
    sources = []
    if os.path.exists(FAISS_INDEX_PATH):
        os.remove(FAISS_INDEX_PATH)
    if os.path.exists(CHUNKS_PATH):
        os.remove(CHUNKS_PATH)
    if os.path.exists(META_PATH):
        os.remove(META_PATH)
    logger.info("Store reset: cleared index and chunks.")


def _get_embeddings(texts: list[str]) -> np.ndarray:
    if embedder is not None:
        try:
            embeddings = embedder.encode(texts, normalize_embeddings=True)
            return np.asarray(embeddings, dtype="float32")
        except Exception:
            logger.exception("Embedding failed; falling back")
            if not FALLBACK_EMBEDDINGS:
                raise
    if not FALLBACK_EMBEDDINGS:
        raise RuntimeError("No embedding model available.")
    return _fallback_embed(texts)


def _rebuild_index(all_chunks: list[str]) -> None:
    global index
    embeddings = _get_embeddings(all_chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)


def add_chunks(new_chunks: list[str], source: str) -> int:
    global index, chunks, sources
    if not new_chunks:
        return 0
    all_chunks = chunks + new_chunks
    all_sources = sources + [source] * len(new_chunks)
    if index is None:
        _rebuild_index(all_chunks)
    else:
        embeddings = _get_embeddings(new_chunks)
        if index.d != embeddings.shape[1]:
            logger.warning("Embedding dim changed (%d -> %d). Rebuilding index.", index.d, embeddings.shape[1])
            _rebuild_index(all_chunks)
        else:
            index.add(embeddings)
    chunks = all_chunks
    sources = all_sources
    save_store()
    return len(new_chunks)

tokenizer = None
model = None

if not FAST_MODE:
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=LOCAL_ONLY)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,
            local_files_only=LOCAL_ONLY
        ).to(DEVICE)

        if os.path.exists(LORA_PATH):
            model = PeftModel.from_pretrained(base_model, LORA_PATH)
        else:
            model = base_model
        model.config.use_cache = True
        model.eval()
    except Exception:
        logger.warning("Generation model not available locally (LOCAL_ONLY=%s)", LOCAL_ONLY)
        tokenizer = None
        model = None

_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def _answer_impl(question: str) -> dict:
    """
    End-to-end RAG:
    - embed query
    - retrieve top-K chunks
    - generate grounded answer
    - return answer + sources
    """

    start = time.time()
    logger.info("Answering question: %s", question)

    # 1️⃣ Embed question
    if index is None or not chunks:
        return {
            "answer": "No documents indexed yet. Please ingest a URL first.",
            "sources": []
        }

    query_embedding = _get_embeddings([question])
    logger.info("Embedding done in %.2fs", time.time() - start)

    # 2️⃣ Retrieve chunks
    if hasattr(index, "d") and index.d != query_embedding.shape[1]:
        logger.warning("Index dimension %s does not match query dim %s", index.d, query_embedding.shape[1])
        return {
            "answer": "Context index is out of sync. Please reset and re-ingest your text.",
            "sources": []
        }
    try:
        scores, indices = index.search(query_embedding, TOP_K)
    except Exception:
        logger.exception("FAISS search failed")
        return {
            "answer": "Search failed. Please reset and re-ingest your text.",
            "sources": []
        }
    retrieved = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        if score < RAG_MIN_SCORE:
            continue
        retrieved.append((chunks[idx], sources[idx] if idx < len(sources) else "", float(score)))
    if not retrieved:
        return {
            "answer": "I don't know. The answer isn't in the provided context.",
            "sources": []
        }
    retrieved_chunks = [c for c, _, _ in retrieved]
    logger.info("Retrieved %d chunks", len(retrieved_chunks))

    # 3️⃣ Build context
    raw_context = "\n".join([str(c) for c in retrieved_chunks])
    context = raw_context[:CONTEXT_CHARS].strip()
    if not context or len(context.split()) < 10:
        return {
            "answer": "I don't know. The answer isn't in the provided context.",
            "sources": [{"text": c, "source": s} for c, s, _ in retrieved]
        }

    # 4️⃣ Prompt (anti-hallucination)
    prompt = (
        "You are a precise assistant.\n"
        "Use ONLY the context below to answer the question.\n"
        "If the answer is not in the context, say: I don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    if FORCE_EXTRACTIVE:
        logger.info("FORCE_EXTRACTIVE enabled - returning sentence answer")
        answer_text = _extractive_answer(question, context)
        return {
            "answer": answer_text or "I don't know. The answer isn't in the provided context.",
            "sources": [{"text": c, "source": s} for c, s, _ in retrieved]
        }

    if FAST_MODE or tokenizer is None or model is None:
        logger.info("FAST_MODE enabled or model unavailable - returning summary")
        summary = _summarize_extractively(context)
        return {
            "answer": summary or "I don't know. The answer isn't in the provided context.",
            "sources": [{"text": c, "source": s} for c, s, _ in retrieved]
        }

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # 5️⃣ Generate answer
    try:
        logger.info("Starting generation")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=GEN_MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                max_time=25.0
            )
        answer_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info("Generation completed in %.2fs", time.time() - start)
    except Exception:
        logger.exception("Generation failed or timed out")
        return {
            "answer": (
                "I couldn't generate a response in time, but here are the most "
                "relevant passages I found:\n\n"
                f"{context}"
            ),
            "sources": [{"text": c, "source": s} for c, s, _ in retrieved]
        }

    return {
        "answer": answer_text.strip(),
        "sources": [{"text": c, "source": s} for c, s, _ in retrieved]
    }


def answer(question: str) -> dict:
    if REQUEST_TIMEOUT <= 0:
        return _answer_impl(question)

    future = _EXECUTOR.submit(_answer_impl, question)
    try:
        return future.result(timeout=REQUEST_TIMEOUT)
    except concurrent.futures.TimeoutError:
        logger.warning("Request timed out after %.1fs", REQUEST_TIMEOUT)
        return {
            "answer": (
                "The model is taking too long. Here are the most relevant "
                "passages I found:\n\n"
                "Please try again with a shorter question."
            ),
            "sources": []
        }
