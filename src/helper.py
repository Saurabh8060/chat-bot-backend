import logging
import os
import re
import importlib.metadata as importlib_metadata
from typing import Any
from typing import List, Tuple

import numpy as np

from src.prompt import INTENT_ROUTER_PROMPT, RAG_PROMPT

# Reduce threading issues on macOS before model loading.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "5"))
CONTEXT_TOP_K = int(os.getenv("CONTEXT_TOP_K", str(TOP_K)))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "2400"))
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.35"))
FALLBACK_ANSWER = "I don't know based on the available medical data."
GREETING_ANSWER = "Hello. Ask me a medical question."
DATASET_NAME = os.getenv("DATASET_NAME", "keivalya/MedQuad-MedicalQnADataset")
DATASET_URL = os.getenv(
    "DATASET_URL",
    f"https://huggingface.co/datasets/{DATASET_NAME}",
)
DATA_SOURCE_ANSWER = (
    f"I answer from the {DATASET_NAME} dataset for this project. Dataset link: {DATASET_URL}"
)
IDENTITY_OR_HELP_ANSWER = (
    "I am a medical information assistant for this project. Ask me about a disease, symptom, cause, diagnosis, or treatment."
)
INTENT_LABELS = ("greeting", "identity_or_help", "data_source", "medical", "unsupported")

HF_MODEL = os.getenv("HF_MODEL", "google/flan-t5-base")
HF_MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", "160"))
HF_MIN_NEW_TOKENS = int(os.getenv("HF_MIN_NEW_TOKENS", "48"))
HF_NUM_BEAMS = int(os.getenv("HF_NUM_BEAMS", "2"))
HF_NO_REPEAT_NGRAM_SIZE = int(os.getenv("HF_NO_REPEAT_NGRAM_SIZE", "3"))
HF_REPETITION_PENALTY = float(os.getenv("HF_REPETITION_PENALTY", "1.15"))
USE_GENERATOR = os.getenv("USE_GENERATOR", "1") == "1"
HF_LOCAL_FILES_ONLY = os.getenv("HF_LOCAL_FILES_ONLY", "0") == "1"
MAX_ANSWER_SENTENCES = int(os.getenv("MAX_ANSWER_SENTENCES", "6"))
MAX_ANSWER_CHARS = int(os.getenv("MAX_ANSWER_CHARS", "900"))
ROUTER_MAX_NEW_TOKENS = int(os.getenv("ROUTER_MAX_NEW_TOKENS", "6"))
ROUTER_MIN_NEW_TOKENS = int(os.getenv("ROUTER_MIN_NEW_TOKENS", "1"))

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST", "")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "medquad")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "")
PINECONE_REGION = os.getenv("PINECONE_REGION", "")
PINECONE_CREATE_INDEX = os.getenv("PINECONE_CREATE_INDEX", "0") == "1"
PINECONE_CONNECT_TIMEOUT = float(os.getenv("PINECONE_CONNECT_TIMEOUT", "3"))
PINECONE_READ_TIMEOUT = float(os.getenv("PINECONE_READ_TIMEOUT", "8"))

_embedder = None
_generator = None
_pc = None
_pc_index = None
logger = logging.getLogger(__name__)


class _EmptyEntryPoints(dict):
    def select(self, **params):
        return []


def _patch_importlib_metadata() -> None:
    if getattr(importlib_metadata, "_rag_chatbot_patched", False):
        return

    # Avoid expensive package/entry-point scans triggered by transformers/torch startup.
    importlib_metadata.packages_distributions = lambda: {}
    importlib_metadata.entry_points = lambda *args, **kwargs: _EmptyEntryPoints()
    importlib_metadata._rag_chatbot_patched = True


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms


def _normalize_question(text: str) -> str:
    text = text.lower().replace("?", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _clean_text(text: str) -> str:
    text = text.replace("Key Points", "").strip()
    text = re.sub(r"^\s*[-•]+\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_topic(text: str) -> str:
    topic = _normalize_question(text)
    topic = re.sub(
        r"^(what is|what are|tell me about|explain|define|describe|information about|do you have information about)\s+",
        "",
        topic,
    )
    topic = re.sub(r"^(a|an|the)\s+", "", topic)
    topic = re.sub(r"[^\w\s-]", " ", topic)
    topic = re.sub(r"\s+", " ", topic).strip()
    return topic


def _resolve_context_topic(
    last_topic: str | None,
    previous_user_message: str | None,
    previous_assistant_message: str | None,
) -> str | None:
    for candidate in (last_topic, previous_user_message, previous_assistant_message):
        topic = _extract_topic(candidate or "")
        if topic:
            return topic
    return None


def _question_depends_on_context(normalized_question: str) -> bool:
    if not normalized_question:
        return False

    if re.search(r"\b(it|them|they|this|that|these|those)\b", normalized_question):
        return True

    vague_followups = (
        "what is the cure",
        "is there a cure",
        "what is the treatment",
        "what are the treatments",
        "how to treat",
        "how do i treat",
        "how can i treat",
        "how to reduce",
        "how do i reduce",
        "how can i reduce",
        "what causes",
        "what are the causes",
        "what causes this",
        "what causes them",
        "what are the symptoms",
        "how to prevent",
        "how do i prevent",
        "how can i prevent",
        "how to manage",
        "how do i manage",
        "how can i manage",
    )
    return normalized_question.startswith(vague_followups)


def _rewrite_followup_question(
    question: str,
    topic: str | None,
) -> tuple[str, bool]:
    if not topic:
        return question.strip(), False

    normalized_question = _normalize_question(question)
    if not normalized_question or not topic:
        return question.strip(), False

    if topic in normalized_question:
        return question.strip(), False

    pronoun_patterns = [
        (r"\bthem\b", topic),
        (r"\bthey\b", topic),
        (r"\bit\b", topic),
        (r"\bthis condition\b", topic),
        (r"\bthis disease\b", topic),
        (r"\bthis\b", topic),
        (r"\bthat\b", topic),
        (r"\bthose\b", topic),
        (r"\bthese\b", topic),
    ]
    rewritten = normalized_question
    replaced = False
    for pattern, replacement in pronoun_patterns:
        next_rewritten, count = re.subn(pattern, replacement, rewritten, count=1)
        if count:
            rewritten = next_rewritten
            return rewritten, True

    if _question_depends_on_context(normalized_question):
        return f"{normalized_question} for {topic}", True

    return question.strip(), False


def _topic_keywords(topic: str) -> list[str]:
    generic_terms = {
        "disease",
        "disorder",
        "condition",
        "syndrome",
        "symptom",
        "symptoms",
        "treatment",
        "treatments",
        "cause",
        "causes",
        "cure",
        "medical",
        "information",
    }
    return [part for part in topic.split() if len(part) >= 4 and part not in generic_terms]


def _match_mentions_topic(topic: str, match: dict) -> bool:
    normalized_topic = _normalize_question(topic)
    question = _normalize_question(match.get("question") or "")
    answer = _normalize_question(match.get("answer") or "")
    if normalized_topic and (normalized_topic in question or normalized_topic in answer):
        return True

    keywords = _topic_keywords(normalized_topic)
    if not keywords:
        return False
    return any(keyword in question or keyword in answer for keyword in keywords)


def _classify_intent_heuristic(question: str) -> str | None:
    normalized = _normalize_question(question)
    if not normalized:
        return None

    if normalized in {
        "hi",
        "hello",
        "hey",
        "hello there",
        "hey there",
        "good morning",
        "good afternoon",
        "good evening",
    }:
        return "greeting"

    if re.fullmatch(r"(hi|hello|hey)\s+[a-z]+", normalized):
        return "greeting"

    if normalized in {
        "who are you",
        "what are you",
        "what can you do",
        "help",
        "can you help me",
        "how can you help",
        "what do you do",
    }:
        return "identity_or_help"

    if normalized in {
        "what data are you based on",
        "what information are you based on",
        "what source are you based on",
        "what sources do you use",
        "what data do you use",
        "what information do you use",
        "where do you get your information",
    }:
        return "data_source"

    return None


def _response_for_intent(intent: str) -> str:
    if intent == "greeting":
        return GREETING_ANSWER
    if intent == "identity_or_help":
        return IDENTITY_OR_HELP_ANSWER
    if intent == "data_source":
        return DATA_SOURCE_ANSWER
    return FALLBACK_ANSWER


def _load_embedder() -> Any:
    global _embedder
    if _embedder is None:
        _patch_importlib_metadata()
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(EMBEDDING_MODEL, local_files_only=HF_LOCAL_FILES_ONLY)
    return _embedder


def _get_generator():
    global _generator
    if _generator is None:
        _patch_importlib_metadata()
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        try:
            import torch

            torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))
            torch.set_num_interop_threads(int(os.getenv("TORCH_INTEROP_THREADS", "1")))
        except Exception:
            pass
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, local_files_only=HF_LOCAL_FILES_ONLY)
        model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL, local_files_only=HF_LOCAL_FILES_ONLY)
        _generator = (tokenizer, model)
    return _generator


def _generate_text(
    prompt: str,
    *,
    max_new_tokens: int | None = None,
    min_new_tokens: int | None = None,
) -> str:
    tokenizer, model = _get_generator()
    inputs = tokenizer(prompt, return_tensors="pt")
    resolved_max_new_tokens = max_new_tokens or HF_MAX_NEW_TOKENS
    resolved_min_new_tokens = HF_MIN_NEW_TOKENS if min_new_tokens is None else min_new_tokens

    kwargs = {
        **inputs,
        "max_new_tokens": resolved_max_new_tokens,
        "min_new_tokens": min(resolved_min_new_tokens, resolved_max_new_tokens),
        "do_sample": False,
        "num_beams": HF_NUM_BEAMS,
        "repetition_penalty": HF_REPETITION_PENALTY,
        "early_stopping": True,
    }
    if HF_NO_REPEAT_NGRAM_SIZE > 0:
        kwargs["no_repeat_ngram_size"] = HF_NO_REPEAT_NGRAM_SIZE

    try:
        import torch

        with torch.inference_mode():
            outputs = model.generate(**kwargs)
    except Exception:
        outputs = model.generate(**kwargs)

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def _generate_answer(question: str, context: str) -> str:
    prompt = RAG_PROMPT.format(context=context, question=question)
    return _finalize_answer(_generate_text(prompt))


def _classify_intent(question: str) -> str:
    prompt = INTENT_ROUTER_PROMPT.format(question=question)
    raw_label = _normalize_question(
        _generate_text(
            prompt,
            max_new_tokens=ROUTER_MAX_NEW_TOKENS,
            min_new_tokens=ROUTER_MIN_NEW_TOKENS,
        )
    )
    canonical_label = re.sub(r"[^a-z]+", "_", raw_label).strip("_")
    for label in INTENT_LABELS:
        if canonical_label == label:
            return label
    for label in INTENT_LABELS:
        if label in canonical_label:
            return label
    logger.warning("Intent router returned unexpected label: %s", raw_label)
    return "unsupported"


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

    total = 0
    batch_size = 200
    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i : i + batch_size]
        batch_vectors = vectors[i : i + batch_size]
        ids = [f"medquad-{i + j}" for j in range(len(batch_pairs))]
        metadata = [{"question": q, "answer": a} for q, a in batch_pairs]
        payload = list(zip(ids, batch_vectors.tolist(), metadata))
        index.upsert(vectors=payload, namespace=PINECONE_NAMESPACE)
        total += len(batch_pairs)

    return total


def retrieve_matches(query: str, k: int | None = None) -> list[dict]:
    embedder = _load_embedder()
    vector = embedder.encode([query], convert_to_numpy=True, show_progress_bar=False).astype("float32")
    vector = _normalize_vectors(vector)
    index = _get_pinecone_index()
    try:
        result = index.query(
            vector=vector[0].tolist(),
            top_k=k or TOP_K,
            include_metadata=True,
            namespace=PINECONE_NAMESPACE,
            _request_timeout=(PINECONE_CONNECT_TIMEOUT, PINECONE_READ_TIMEOUT),
        )
    except Exception as exc:
        logger.warning("Pinecone query failed: %s", exc)
        return []

    matches: list[dict] = []
    for match in result.get("matches", []):
        metadata = match.get("metadata") or {}
        matches.append(
            {
                "score": float(match.get("score", 0.0)),
                "question": metadata.get("question"),
                "answer": metadata.get("answer"),
            }
        )
    return matches


def _build_context(matches: list[dict]) -> str:
    snippets: list[str] = []
    seen_questions: set[str] = set()

    for index, match in enumerate(matches[:CONTEXT_TOP_K], start=1):
        question = _clean_text(match.get("question") or "")
        answer = _clean_text(match.get("answer") or "")
        if not question or not answer:
            continue

        normalized = _normalize_question(question)
        if normalized in seen_questions:
            continue
        seen_questions.add(normalized)
        snippets.append(f"[{index}] Q: {question}\nA: {answer}")

    context = "\n\n".join(snippets).strip()
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS].rstrip()
    return context


def _extractive_answer(matches: list[dict]) -> str:
    if not matches:
        return ""
    return _clean_text(matches[0].get("answer") or "")


def _shorten_answer(answer: str) -> str:
    answer = _clean_text(answer)
    if not answer:
        return ""

    sentence_candidates = re.split(r"(?<=[.!?])\s+", answer)
    sentences = [part.strip() for part in sentence_candidates if part.strip()]
    if sentences:
        shortened = " ".join(sentences[:MAX_ANSWER_SENTENCES]).strip()
    else:
        shortened = answer

    if len(shortened) > MAX_ANSWER_CHARS:
        clipped = shortened[:MAX_ANSWER_CHARS].rstrip()
        if " " in clipped:
            clipped = clipped.rsplit(" ", 1)[0]
        shortened = clipped.rstrip(" ,;:") + "..."

    return shortened


def _finalize_answer(answer: str) -> str:
    answer = _clean_text(answer)
    if not answer:
        return FALLBACK_ANSWER
    lowered = answer.lower()
    if lowered.startswith("i don't know") or lowered.startswith("i do not know"):
        return FALLBACK_ANSWER
    return _shorten_answer(answer)


def dedupe_pairs(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen: dict[str, Tuple[str, str]] = {}
    for question, answer in pairs:
        normalized = _normalize_question(question)
        if not normalized:
            continue
        if normalized not in seen:
            seen[normalized] = (_clean_text(question), _clean_text(answer))
    return list(seen.values())


def answer_question(
    question: str,
    last_topic: str | None = None,
    previous_user_message: str | None = None,
    previous_assistant_message: str | None = None,
    debug: bool = False,
) -> dict:
    question = question.strip()
    if not question:
        return {"answer": "Please provide a question.", "source_question": None}

    heuristic_intent = _classify_intent_heuristic(question)
    if heuristic_intent:
        response = {
            "answer": _response_for_intent(heuristic_intent),
            "source_question": None,
        }
        if debug:
            response["matches"] = []
            response["context"] = ""
            response["intent"] = heuristic_intent
        return response

    context_topic = _resolve_context_topic(
        last_topic,
        previous_user_message,
        previous_assistant_message,
    )
    resolved_question, topic_locked = _rewrite_followup_question(question, context_topic)
    normalized_question = _normalize_question(resolved_question)
    matches = retrieve_matches(normalized_question, k=TOP_K)
    if topic_locked and context_topic:
        matches = [match for match in matches if _match_mentions_topic(context_topic, match)]
    if not matches:
        intent = "unsupported"
        if USE_GENERATOR:
            try:
                intent = _classify_intent(resolved_question)
            except Exception as exc:
                logger.warning("Intent routing failed after empty retrieval: %s", exc)
        response = {
            "answer": _response_for_intent(intent),
            "source_question": None,
        }
        if debug:
            response["matches"] = []
            response["context"] = ""
            response["intent"] = intent
        return response

    best_match = matches[0]
    best_score = float(best_match.get("score", 0.0))
    if best_score < SIM_THRESHOLD:
        intent = "unsupported"
        if USE_GENERATOR:
            try:
                intent = _classify_intent(resolved_question)
            except Exception as exc:
                logger.warning("Intent routing failed below similarity threshold: %s", exc)
        response = {
            "answer": _response_for_intent(intent),
            "source_question": None,
        }
        if debug:
            response["matches"] = matches
            response["context"] = ""
            response["intent"] = intent
        return response

    intent = "medical"
    context = _build_context(matches)
    if not context:
        return {
            "answer": FALLBACK_ANSWER,
            "source_question": None,
            **({"matches": matches} if debug else {}),
        }

    if USE_GENERATOR:
        try:
            answer = _generate_answer(resolved_question, context)
        except Exception as exc:
            logger.warning("Generator failed, falling back to extractive answer: %s", exc)
            answer = _finalize_answer(_extractive_answer(matches))
    else:
        answer = _finalize_answer(_extractive_answer(matches))

    response = {
        "answer": answer,
        "source_question": best_match.get("question"),
    }
    if debug:
        response["matches"] = matches
        response["context"] = context
        response["intent"] = intent
        response["resolved_question"] = resolved_question
        response["last_topic"] = last_topic
        response["context_topic"] = context_topic
        response["topic_locked"] = topic_locked
    return response
