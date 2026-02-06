import json
import os
import re
from pathlib import Path
from typing import List, Tuple

# Reduce OpenMP/threading issues on macOS by setting defaults before heavy imports.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INDEX_PATH = DATA_DIR / "faiss.index"
META_PATH = DATA_DIR / "medquad_meta.json"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "5"))
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.35"))
HF_MODEL = os.getenv("HF_MODEL", "google/flan-t5-base")
HF_MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", "256"))
USE_GENERATOR = os.getenv("USE_GENERATOR", "1") == "1"

_embedder = None
_index = None
_meta = None
_generator = None


def _load_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms


def _load_index() -> None:
    global _index, _meta
    if _index is not None and _meta is not None:
        return
    if INDEX_PATH.exists() and META_PATH.exists():
        _index = faiss.read_index(str(INDEX_PATH))
        _meta = json.loads(META_PATH.read_text())
    else:
        _index = None
        _meta = None


def _save_index(index: faiss.Index, meta: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2))


def build_index(pairs: List[Tuple[str, str]]) -> int:
    embedder = _load_embedder()
    texts = [f"{q}\n{a}" for q, a in pairs]
    vectors = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype("float32")
    vectors = _normalize_vectors(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    meta = [{"question": q, "answer": a} for q, a in pairs]
    _save_index(index, meta)
    # refresh globals
    global _index, _meta
    _index = index
    _meta = meta
    return len(meta)


def retrieve_context(query: str, k: int | None = None) -> str:
    _load_index()
    if _index is None or _meta is None:
        return ""
    embedder = _load_embedder()
    vec = embedder.encode([query], convert_to_numpy=True, show_progress_bar=False).astype("float32")
    vec = _normalize_vectors(vec)
    k = k or TOP_K
    scores, idxs = _index.search(vec, k)
    ctx = []
    for idx in idxs[0]:
        if idx < 0 or idx >= len(_meta):
            continue
        item = _meta[idx]
        ctx.append(f"{item['question']}\n{item['answer']}")
    return "\n\n".join(ctx).strip()


def retrieve_matches(query: str, k: int | None = None) -> list[dict]:
    _load_index()
    if _index is None or _meta is None:
        return []
    embedder = _load_embedder()
    vec = embedder.encode([query], convert_to_numpy=True, show_progress_bar=False).astype("float32")
    vec = _normalize_vectors(vec)
    k = k or TOP_K
    scores, idxs = _index.search(vec, k)
    matches: list[dict] = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0 or idx >= len(_meta):
            continue
        item = _meta[idx]
        matches.append(
            {
                "score": float(score),
                "question": item.get("question"),
                "answer": item.get("answer"),
            }
        )
    return matches


def _get_generator():
    global _generator
    if _generator is None:
        try:
            import torch
            torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))
            torch.set_num_interop_threads(int(os.getenv("TORCH_INTEROP_THREADS", "1")))
        except Exception:
            # Best-effort; if torch isn't available, imports below will fail anyway.
            pass
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL)
        _generator = (tokenizer, model)
    return _generator


def _generate_text(prompt: str) -> str:
    tokenizer, model = _get_generator()
    inputs = tokenizer(prompt, return_tensors="pt")
    try:
        import torch
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=HF_MAX_NEW_TOKENS)
    except Exception:
        outputs = model.generate(**inputs, max_new_tokens=HF_MAX_NEW_TOKENS)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def _is_unsafe_medical_query(query: str) -> bool:
    q = query.lower()
    blocked = [
        "diagnose", "diagnosis", "prescribe", "prescription", "dose", "dosage",
        "emergency", "urgent",
        "should i take", "what should i take", "what medicine", "what drug",
    ]
    return any(term in q for term in blocked)


def _needs_disclaimer(query: str) -> bool:
    q = query.lower()
    terms = [
        "treatment", "treat", "therapy", "medicine", "medication",
        "drug", "side effect", "dosage", "dose", "cure",
    ]
    return any(term in q for term in terms)


def _append_disclaimer(answer: str) -> str:
    disclaimer = (
        " This is general information only and not medical advice. "
        "For diagnosis or treatment decisions, consult a qualified healthcare professional."
    )
    if disclaimer.strip() in answer:
        return answer
    return f"{answer}{disclaimer}"


def _prompt(answer: str, question: str) -> str:
    return (
        "You are a medical assistant.\n"
        "Answer the question using ONLY the answer provided.\n"
        "Do not add or infer any medical information.\n\n"
        f"Answer:\n{answer}\n\n"
        f"Question:\n{question}\n"
    )


def _normalize_question(q: str) -> str:
    q = q.lower()
    q = q.replace("(are)", "")
    q = q.replace("?", "")
    q = re.sub(r"\s+", " ", q).strip()
    return q


def _clean_answer(text: str) -> str:
    text = text.replace("Key Points", "")
    text = text.strip()
    # Remove leading bullets/indents
    text = re.sub(r"^\s*[-•]+\s*", "", text)
    # Keep only first meaningful section and truncate
    text = re.sub(r"\s+", " ", text).strip()
    return text


def dedupe_pairs(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen = {}
    for q, a in pairs:
        norm_q = _normalize_question(q)
        if not norm_q:
            continue
        if norm_q not in seen:
            seen[norm_q] = (q.strip(), _clean_answer(a))
    return list(seen.values())


def answer_question(question: str, debug: bool = False) -> dict:
    if not question:
        return {"answer": "Please provide a question.", "source_question": None}

    if _is_unsafe_medical_query(question):
        return {
            "answer": "I cannot provide diagnosis or treatment. Please consult a medical professional.",
            "source_question": None,
        }

    _load_index()
    if _index is None or _meta is None:
        return {
            "answer": "I don't know based on the available medical data.",
            "source_question": None,
            **({"matches": []} if debug else {}),
        }

    embedder = _load_embedder()
    vec = embedder.encode([question], convert_to_numpy=True, show_progress_bar=False).astype("float32")
    vec = _normalize_vectors(vec)
    scores, idxs = _index.search(vec, TOP_K)
    best_score = float(scores[0][0]) if len(scores[0]) else 0.0
    best_idx = int(idxs[0][0]) if len(idxs[0]) else -1
    if best_idx < 0 or best_idx >= len(_meta) or best_score < SIM_THRESHOLD:
        return {
            "answer": "I don't know based on the available medical data.",
            "source_question": None,
            **({"matches": retrieve_matches(question)} if debug else {}),
        }

    item = _meta[best_idx]
    answer_text = item["answer"]
    if not USE_GENERATOR:
        answer_out = _append_disclaimer(answer_text) if _needs_disclaimer(question) else answer_text
        return {
            "answer": answer_out,
            "source_question": item["question"],
            **({"matches": retrieve_matches(question)} if debug else {}),
        }
    prompt = _prompt(answer_text, question)
    result = _generate_text(prompt)

    if "I don't know based on the available medical data." in result:
        answer = "I don't know based on the available medical data."
    else:
        answer = _append_disclaimer(result) if _needs_disclaimer(question) else result

    return {
        "answer": answer,
        "source_question": item["question"],
        **({"matches": retrieve_matches(question)} if debug else {}),
    }
