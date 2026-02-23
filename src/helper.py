import os
import re
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Reduce OpenMP/threading issues on macOS by setting defaults before heavy imports.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "5"))
RERANK_CANDIDATE_K = int(os.getenv("RERANK_CANDIDATE_K", str(max(20, TOP_K * 8))))
CONTEXT_TOP_K = int(os.getenv("CONTEXT_TOP_K", str(TOP_K)))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "2400"))
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.35"))
FALLBACK_ANSWER = "I don't know based on the available medical data."

HF_MODEL = os.getenv("HF_MODEL", "google/flan-t5-base")
HF_MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", "256"))
HF_MIN_NEW_TOKENS = int(os.getenv("HF_MIN_NEW_TOKENS", "24"))
HF_NUM_BEAMS = int(os.getenv("HF_NUM_BEAMS", "3"))
HF_NO_REPEAT_NGRAM_SIZE = int(os.getenv("HF_NO_REPEAT_NGRAM_SIZE", "3"))
HF_REPETITION_PENALTY = float(os.getenv("HF_REPETITION_PENALTY", "1.15"))
USE_GENERATOR = os.getenv("USE_GENERATOR", "1") == "1"

USE_CROSS_ENCODER = os.getenv("USE_CROSS_ENCODER", "1") == "1"
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST", "")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "medquad")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "")
PINECONE_REGION = os.getenv("PINECONE_REGION", "")
PINECONE_CREATE_INDEX = os.getenv("PINECONE_CREATE_INDEX", "0") == "1"
BOT_NAME = os.getenv("BOT_NAME", "Medical Knowledge Assistant")
DATA_SOURCE_LABEL = os.getenv("DATA_SOURCE_LABEL", "MedQuAD medical Q&A dataset")

_embedder = None
_reranker = None
_generator = None
_pc = None
_pc_index = None

_TREATMENT_TERMS = {
    "treat",
    "treatment",
    "therapy",
    "therapies",
    "manage",
    "management",
    "cure",
    "medication",
    "medicine",
    "drug",
    "remedy",
}
_DEFINITION_TERMS = {
    "what",
    "what is",
    "define",
    "definition",
    "meaning",
    "symptom",
    "symptoms",
    "cause",
    "causes",
}
_GENERIC_QUERY_TERMS = {
    "how",
    "what",
    "when",
    "where",
    "why",
    "which",
    "who",
    "can",
    "could",
    "should",
    "would",
    "is",
    "are",
    "was",
    "were",
    "to",
    "for",
    "of",
    "the",
    "a",
    "an",
    "and",
    "or",
    "in",
    "on",
    "with",
    "about",
}
_QUERY_NORMALIZATION_MAP = {
    "treatement": "treatment",
    "medecine": "medicine",
}
_GREETING_PATTERNS = [
    r"^\s*(hi|hello|hey|yo|hola)\b[!. ]*$",
    r"^\s*(hi|hello|hey)\s+(there|bot)?\s*[!. ]*$",
]
_STATUS_PATTERNS = [
    r"\bhow are you\b",
    r"\bhow r you\b",
]
_CAPABILITY_PATTERNS = [
    r"\bwhat do you do\b",
    r"\bwhat can you do\b",
    r"\bwho are you\b",
    r"\babout this chatbot\b",
]
_DATA_SOURCE_PATTERNS = [
    r"\bwhich data source\b",
    r"\bwhat data source\b",
    r"\bwhere (does|do) (your|you) data come from\b",
    r"\bsource of (your|the) data\b",
    r"\bdataset\b",
]


def _load_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _get_reranker():
    global _reranker
    if _reranker is not None:
        return _reranker
    if not USE_CROSS_ENCODER:
        return None
    try:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(RERANK_MODEL)
        return _reranker
    except Exception:
        return None


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms


def _get_pinecone_index():
    global _pc, _pc_index
    if _pc_index is not None:
        return _pc_index
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is not set")
    try:
        from pinecone import Pinecone
    except Exception as exc:
        raise RuntimeError("pinecone package is not installed") from exc

    _pc = Pinecone(api_key=PINECONE_API_KEY)
    if PINECONE_INDEX_HOST:
        _pc_index = _pc.Index(host=PINECONE_INDEX_HOST)
        return _pc_index

    if not PINECONE_INDEX_NAME:
        raise RuntimeError("PINECONE_INDEX_NAME is not set")

    if PINECONE_CREATE_INDEX:
        if not (PINECONE_CLOUD and PINECONE_REGION):
            raise RuntimeError("PINECONE_CLOUD and PINECONE_REGION are required to create an index")
        existing = {idx["name"] for idx in _pc.list_indexes()}
        if PINECONE_INDEX_NAME not in existing:
            pass

    _pc_index = _pc.Index(name=PINECONE_INDEX_NAME)
    return _pc_index


def build_index(pairs: List[Tuple[str, str]]) -> int:
    embedder = _load_embedder()
    texts = [f"{q}\n{a}" for q, a in pairs]
    vectors = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype("float32")
    vectors = _normalize_vectors(vectors)
    index = _get_pinecone_index()

    if PINECONE_CREATE_INDEX and PINECONE_CLOUD and PINECONE_REGION:
        from pinecone import ServerlessSpec

        existing = {idx["name"] for idx in _pc.list_indexes()}
        if PINECONE_INDEX_NAME and PINECONE_INDEX_NAME not in existing:
            _pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=vectors.shape[1],
                metric="cosine",
                spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
            )
            index = _pc.Index(name=PINECONE_INDEX_NAME)

    batch_size = 200
    total = 0
    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i : i + batch_size]
        batch_vecs = vectors[i : i + batch_size]
        ids = [f"medquad-{i + j}" for j in range(len(batch_pairs))]
        metadata = [{"question": q, "answer": a} for q, a in batch_pairs]
        to_upsert = list(zip(ids, batch_vecs.tolist(), metadata))
        index.upsert(vectors=to_upsert, namespace=PINECONE_NAMESPACE)
        total += len(batch_pairs)
    return total


def retrieve_matches(query: str, k: int | None = None) -> list[dict]:
    embedder = _load_embedder()
    vec = embedder.encode([query], convert_to_numpy=True, show_progress_bar=False).astype("float32")
    vec = _normalize_vectors(vec)
    k = k or TOP_K
    index = _get_pinecone_index()
    res = index.query(
        vector=vec[0].tolist(),
        top_k=k,
        include_metadata=True,
        namespace=PINECONE_NAMESPACE,
    )
    matches: list[dict] = []
    for match in res.get("matches", []):
        meta = match.get("metadata") or {}
        matches.append(
            {
                "score": float(match.get("score", 0.0)),
                "question": meta.get("question"),
                "answer": meta.get("answer"),
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
            outputs = model.generate(
                **inputs,
                max_new_tokens=HF_MAX_NEW_TOKENS,
                min_new_tokens=min(HF_MIN_NEW_TOKENS, HF_MAX_NEW_TOKENS),
                do_sample=False,
                num_beams=HF_NUM_BEAMS,
                no_repeat_ngram_size=HF_NO_REPEAT_NGRAM_SIZE,
                repetition_penalty=HF_REPETITION_PENALTY,
                early_stopping=True,
            )
    except Exception:
        outputs = model.generate(
            **inputs,
            max_new_tokens=HF_MAX_NEW_TOKENS,
            min_new_tokens=min(HF_MIN_NEW_TOKENS, HF_MAX_NEW_TOKENS),
            do_sample=False,
            num_beams=HF_NUM_BEAMS,
            no_repeat_ngram_size=HF_NO_REPEAT_NGRAM_SIZE,
            repetition_penalty=HF_REPETITION_PENALTY,
            early_stopping=True,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def _is_unsafe_medical_query(query: str) -> bool:
    q = query.lower()
    blocked = [
        "diagnose",
        "diagnosis",
        "prescribe",
        "prescription",
        "dose",
        "dosage",
        "emergency",
        "urgent",
        "should i take",
        "what should i take",
        "what medicine",
        "what drug",
    ]
    return any(term in q for term in blocked)


def _needs_disclaimer(query: str) -> bool:
    q = query.lower()
    terms = [
        "treatment",
        "treat",
        "therapy",
        "medicine",
        "medication",
        "drug",
        "side effect",
        "dosage",
        "dose",
        "cure",
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


def _strip_fallback_prefix(text: str) -> str:
    if not text:
        return text
    t = text.strip()
    low = t.lower()
    fb = FALLBACK_ANSWER.lower()
    if low.startswith(fb):
        remainder = t[len(FALLBACK_ANSWER) :].strip(" \n\t.:;-")
        if remainder:
            return remainder
        return FALLBACK_ANSWER
    return t


def _dedupe_adjacent_csv_segments(text: str) -> str:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) <= 1:
        return text
    deduped: list[str] = []
    for p in parts:
        if not p:
            continue
        if deduped and deduped[-1].lower() == p.lower():
            continue
        deduped.append(p)
    return ", ".join(deduped)


def _clean_generated_answer(text: str) -> str:
    cleaned = _strip_fallback_prefix(text)
    cleaned = _dedupe_adjacent_csv_segments(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _postprocess_answer(answer: str) -> str:
    return _clean_generated_answer(answer)


def _prompt(context: str, question: str) -> str:
    return (
        "You are a medical assistant.\n"
        "Answer the question using ONLY the provided context.\n"
        "Do not add or infer any medical information.\n"
        "Write natural, clear English with complete sentences.\n"
        "Be concise and practical: 3-6 sentences maximum.\n"
        "Do not repeat the same phrase, citation, or source name.\n"
        "If the context is insufficient, answer exactly: "
        "I don't know based on the available medical data.\n\n"
        f"Context:\n{context}\n\n"
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
    text = re.sub(r"^\s*[-•]+\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(?:\bsee\b\s*)+$", "", text, flags=re.IGNORECASE).strip(" -,:;")
    return text


def _tokenize(text: str) -> set[str]:
    return {t for t in re.split(r"\W+", text.lower()) if len(t) > 2}


def _has_term(text: str, terms: set[str]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def _normalize_query_text(query: str) -> str:
    normalized = query.strip().lower()
    for src, dst in _QUERY_NORMALIZATION_MAP.items():
        normalized = re.sub(rf"\b{re.escape(src)}\b", dst, normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _query_intent(query: str) -> str:
    if _has_term(query, _TREATMENT_TERMS):
        return "treatment"
    if _has_term(query, _DEFINITION_TERMS):
        return "definition"
    return "other"


def _match_any_pattern(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def _handle_non_medical_query(question: str) -> str | None:
    q = question.strip()
    if not q:
        return None
    if _match_any_pattern(q, _GREETING_PATTERNS):
        return f"Hi! I am {BOT_NAME}. You can ask me medical information questions."
    if _match_any_pattern(q, _STATUS_PATTERNS):
        return "I am doing well. How can I help you with medical information today?"
    if _match_any_pattern(q, _CAPABILITY_PATTERNS):
        return (
            f"I am {BOT_NAME}. I answer medical information questions by retrieving relevant "
            "content from a curated medical Q&A knowledge base."
        )
    if _match_any_pattern(q, _DATA_SOURCE_PATTERNS):
        return (
            f"My primary data source is {DATA_SOURCE_LABEL}. "
            "I retrieve relevant entries and generate an answer from that context."
        )
    return None


def _content_tokens(text: str) -> set[str]:
    noisy = _GENERIC_QUERY_TERMS | _TREATMENT_TERMS | _DEFINITION_TERMS
    return {t for t in _tokenize(text) if t not in noisy}


def _rerank_matches(query: str, matches: list[dict]) -> list[dict]:
    if not matches:
        return matches

    q_tokens = _tokenize(query)
    intent = _query_intent(query)
    reranker = _get_reranker()
    ce_scores: list[float] | None = None
    if reranker is not None:
        try:
            pairs = [
                [query, f"{(m.get('question') or '').strip()}\n{(m.get('answer') or '').strip()}"]
                for m in matches
            ]
            ce_scores = [float(x) for x in reranker.predict(pairs)]
        except Exception:
            ce_scores = None

    reranked: list[dict] = []
    for i, m in enumerate(matches):
        source_q = (m.get("question") or "").strip()
        source_tokens = _tokenize(source_q)
        overlap = len(q_tokens & source_tokens)
        adjusted = float(m.get("score", 0.0)) + min(0.15, overlap * 0.03)
        if ce_scores is not None:
            ce = max(-5.0, min(5.0, ce_scores[i]))
            adjusted += 0.06 * ce
        if intent == "treatment":
            adjusted += 0.08 if _has_term(source_q, _TREATMENT_TERMS) else -0.12
        elif intent == "definition":
            adjusted += 0.05 if _has_term(source_q, _DEFINITION_TERMS) else 0.0
        m2 = dict(m)
        m2["adjusted_score"] = adjusted
        reranked.append(m2)

    reranked.sort(key=lambda x: float(x.get("adjusted_score", 0.0)), reverse=True)
    return reranked


def _entity_filtered_matches(query: str, ranked_matches: list[dict]) -> list[dict]:
    q_content = _content_tokens(query)
    if not q_content:
        return ranked_matches
    filtered: list[dict] = []
    for match in ranked_matches:
        src_q = match.get("question") or ""
        src_a = match.get("answer") or ""
        src_tokens = _tokenize(src_q) | _tokenize(src_a)
        if q_content & src_tokens:
            filtered.append(match)
    return filtered


def _build_context(matches: list[dict]) -> str:
    snippets: list[str] = []
    seen_questions: set[str] = set()
    for i, match in enumerate(matches[:CONTEXT_TOP_K], start=1):
        q = (match.get("question") or "").strip()
        a = _clean_answer(match.get("answer") or "")
        if not q or not a:
            continue
        key = _normalize_question(q)
        if key in seen_questions:
            continue
        seen_questions.add(key)
        snippets.append(f"[{i}] Q: {q}\nA: {a}")

    context = "\n\n".join(snippets).strip()
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS].rstrip()
    return context


def _extractive_from_matches(matches: list[dict]) -> str:
    context = _build_context(matches)
    if not context:
        return ""
    answer_lines = [line for line in context.splitlines() if line.startswith("A: ")]
    if not answer_lines:
        return ""
    return _strip_fallback_prefix(_clean_answer(answer_lines[0][3:]))


def retrieve_context(query: str, k: int | None = None) -> str:
    normalized_query = _normalize_query_text(query)
    candidate_k = max(k or TOP_K, CONTEXT_TOP_K, RERANK_CANDIDATE_K)
    matches = retrieve_matches(normalized_query, k=candidate_k)
    if not matches:
        return ""
    ranked = _rerank_matches(normalized_query, matches)
    filtered = _entity_filtered_matches(normalized_query, ranked)
    active = filtered or ranked
    return _build_context(active)


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

    non_medical = _handle_non_medical_query(question)
    if non_medical is not None:
        return {
            "answer": non_medical,
            "source_question": None,
            **({"matches": []} if debug else {}),
        }

    if _is_unsafe_medical_query(question):
        return {
            "answer": "I cannot provide diagnosis or treatment. Please consult a medical professional.",
            "source_question": None,
        }

    normalized_question = _normalize_query_text(question)
    candidate_k = max(TOP_K, RERANK_CANDIDATE_K, CONTEXT_TOP_K)
    matches = retrieve_matches(normalized_question, k=candidate_k)
    if not matches:
        return {
            "answer": "I don't know based on the available medical data.",
            "source_question": None,
            **({"matches": []} if debug else {}),
        }

    ranked_matches = _rerank_matches(normalized_question, matches)[:candidate_k]
    entity_matches = _entity_filtered_matches(normalized_question, ranked_matches)
    active_matches = entity_matches or ranked_matches
    if not active_matches:
        return {
            "answer": "I don't know based on the available medical data.",
            "source_question": None,
            **({"matches": ranked_matches} if debug else {}),
        }

    best = active_matches[0]
    best_score = float(best.get("adjusted_score", best.get("score", 0.0)) or 0.0)
    if best_score < SIM_THRESHOLD:
        return {
            "answer": "I don't know based on the available medical data.",
            "source_question": None,
            **({"matches": ranked_matches} if debug else {}),
        }
    if _query_intent(normalized_question) == "treatment":
        best_q = best.get("question") or ""
        best_a = best.get("answer") or ""
        has_treatment_signal = _has_term(best_q, _TREATMENT_TERMS) or _has_term(best_a, _TREATMENT_TERMS)
        if not has_treatment_signal:
            return {
                "answer": "I don't know based on the available medical data.",
                "source_question": None,
                **({"matches": ranked_matches} if debug else {}),
            }

    context = _build_context(active_matches)
    if not context:
        return {
            "answer": "I don't know based on the available medical data.",
            "source_question": None,
            **({"matches": ranked_matches} if debug else {}),
        }

    source_question = best.get("question")
    if not USE_GENERATOR:
        extractive = _extractive_from_matches(active_matches)
        answer_out = extractive or FALLBACK_ANSWER
        answer_out = _postprocess_answer(answer_out)
        if answer_out != FALLBACK_ANSWER and _needs_disclaimer(question):
            answer_out = _append_disclaimer(answer_out)
        return {
            "answer": answer_out,
            "source_question": source_question,
            **({"matches": ranked_matches, "context": context} if debug else {}),
        }

    prompt = _prompt(context, question)
    result = _generate_text(prompt)
    cleaned_result = _postprocess_answer(result)
    if cleaned_result == FALLBACK_ANSWER:
        answer = FALLBACK_ANSWER
    else:
        answer = _append_disclaimer(cleaned_result) if _needs_disclaimer(question) else cleaned_result

    return {
        "answer": answer,
        "source_question": source_question,
        **({"matches": ranked_matches, "context": context} if debug else {}),
    }
