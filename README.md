---
title: Medical Chatbot Backend
emoji: "🩺"
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Medical Chatbot Backend

FastAPI backend for a medical RAG chatbot powered by Pinecone retrieval, Sentence Transformers embeddings, and Hugging Face generation.

This backend retrieves relevant medical Q/A pairs, applies filtering and reranking, and returns either a grounded answer or a safe fallback when the available context is not strong enough.

## Features

- FastAPI API for chat-style question answering
- Pinecone-backed semantic retrieval
- Sentence Transformers embeddings with vector normalization
- Optional cross-encoder reranking for better relevance
- Query typo correction and entity-aware filtering
- Safe fallback behavior for unsupported questions
- Basic medical safety guardrails and disclaimers
- LoRA fine-tuning script for the generator model

## Tech Stack

- FastAPI
- Pydantic
- Sentence Transformers
- Pinecone
- Hugging Face Transformers
- PEFT / LoRA
- NumPy

## Project Structure

```text
.
|-- main.py
|-- requirements.txt
|-- src/
|   |-- helper.py
|   `-- prompt.py
|-- scripts/
|   `-- ingest_medquad.py
|-- training/
|   `-- finetune_lora.py
|-- embeddings/
|   `-- build_index.py
`-- data/
```

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create `backend/.env`

Configure the environment variables your deployment needs.

```env
PINECONE_API_KEY=
PINECONE_INDEX_NAME=
PINECONE_INDEX_HOST=
PINECONE_NAMESPACE=medquad
PINECONE_CLOUD=
PINECONE_REGION=
PINECONE_CREATE_INDEX=0

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
HF_MODEL=google/flan-t5-base
USE_GENERATOR=1
USE_CROSS_ENCODER=1
TOP_K=5
SIM_THRESHOLD=0.35
PORT=8000
```

Notes:

- Set `PINECONE_INDEX_HOST` if you are connecting directly to a hosted Pinecone index.
- Set `PINECONE_INDEX_NAME` if you want to connect by index name.
- Set `PINECONE_CLOUD`, `PINECONE_REGION`, and `PINECONE_CREATE_INDEX=1` if this service should create the Pinecone index automatically.

## Run the API

```bash
uvicorn main:app --reload --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

## API Endpoints

### `GET /health`

Returns:

```json
{ "status": "ok" }
```

### `POST /chat`

Request body:

```json
{
  "message": "What is acne?",
  "question": "What is acne?",
  "debug": true
}
```

Typical response:

```json
{
  "answer": "...",
  "source_question": "...",
  "matches": []
}
```

## How It Works

1. The API receives a user question at `POST /chat`.
2. The question is embedded with `sentence-transformers/all-MiniLM-L6-v2`.
3. The backend retrieves relevant Q/A entries from Pinecone.
4. Matches can be reranked and filtered for stronger entity alignment.
5. The backend either generates a grounded response with `google/flan-t5-base` or returns a safe fallback when confidence is too low.

## Data Ingestion

This repo includes a MedQuAD ingestion script:

```bash
python scripts/ingest_medquad.py
```

Examples:

```bash
python scripts/ingest_medquad.py --limit 200
python scripts/ingest_medquad.py --dataset keivalya/MedQuad-MedicalQnADataset --split train
```

The script deduplicates medical Q/A pairs, embeds them, and uploads them to Pinecone.

## Fine-Tuning

You can fine-tune the generator with LoRA:

```bash
python training/finetune_lora.py
```

Artifacts are saved to `lora_model` by default.

## Troubleshooting

### Pinecone errors

- Confirm `PINECONE_API_KEY` is set.
- Confirm either `PINECONE_INDEX_HOST` or `PINECONE_INDEX_NAME` is configured.
- If auto-creating the index, also set `PINECONE_CLOUD`, `PINECONE_REGION`, and `PINECONE_CREATE_INDEX=1`.

### Slow local inference

- Set `USE_GENERATOR=0` to return extractive answers only.
- Set `USE_CROSS_ENCODER=0` to disable reranking and reduce resource usage.

### Empty or weak answers

- Check whether your Pinecone namespace contains indexed medical Q/A data.
- Lowering `SIM_THRESHOLD` may increase recall, but it can also reduce answer reliability.

## Notes

- This backend uses custom retrieval and generation orchestration rather than LangChain chains.
- It is intended for educational and development use.
- It should not be treated as a substitute for professional medical advice.
