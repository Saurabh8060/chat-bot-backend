import os
import io
from dotenv import load_dotenv

# Load .env before importing helper (helper reads env vars at import time)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader

from src.helper import (
    ensure_vectorstore,
    split_text,
    ingest_text,
    answer_question,
    reset_store,
    PDF_DIR
)

app = FastAPI(title="RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class IngestRequest(BaseModel):
    text: str
    replace: bool | None = False


class ChatRequest(BaseModel):
    message: str | None = None
    question: str | None = None
    last_topic: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.on_event("startup")
def load_pdfs_on_startup():
    # Load PDFs from local directory on startup (optional)
    if os.getenv("ENABLE_PDF_INGEST", "false").lower() == "true" and os.path.isdir(PDF_DIR):
        ensure_vectorstore()
        ingest_pdfs(replace=False)


# NOTE: PDFs are loaded from a local directory. This endpoint is kept for optional
# manual ingestion to avoid breaking existing clients.
@app.post("/ingest")
def ingest(payload: IngestRequest):
    text = payload.text.strip()
    if not text:
        return {"status": "error", "message": "Text is required"}

    ensure_vectorstore()
    chunks = split_text(text)
    if not chunks:
        return {"status": "error", "message": "No usable text found"}

    added = ingest_text(chunks, replace=bool(payload.replace))
    return {"status": "ok", "chunks_added": added}


@app.post("/chat")
def chat(payload: ChatRequest):
    question = (payload.message or payload.question or "").strip()
    if not question:
        return {"answer": "Please provide a question.", "sources": []}

    ensure_vectorstore()
    return answer_question(question, last_topic=payload.last_topic)


@app.post("/ingest_pdf")
async def ingest_pdf(file: UploadFile = File(...), replace: bool = False):
    if not file.filename.lower().endswith(".pdf"):
        return {"status": "error", "message": "Please upload a PDF file."}

    ensure_vectorstore()

    data = await file.read()
    reader = PdfReader(io.BytesIO(data))
    pages_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages_text.append(text.strip())
    full_text = "\n\n".join(pages_text).strip()
    if not full_text:
        return {"status": "error", "message": "No extractable text found in PDF."}

    chunks = split_text(full_text)
    if not chunks:
        return {"status": "error", "message": "No usable text found in PDF."}

    added = ingest_text(chunks, replace=replace)
    return {"status": "ok", "chunks_added": added}


@app.post("/ingest_pdfs")
def ingest_pdfs(replace: bool = False):
    ensure_vectorstore()
    if os.getenv("ENABLE_PDF_INGEST", "false").lower() != "true":
        return {"status": "ok", "message": "PDF ingestion disabled. Using existing Pinecone vectors."}
    if not os.path.isdir(PDF_DIR):
        return {"status": "error", "message": f"PDF_DIR not found: {PDF_DIR}"}

    all_text = []
    for name in os.listdir(PDF_DIR):
        if not name.lower().endswith(".pdf"):
            continue
        path = os.path.join(PDF_DIR, name)
        reader = PdfReader(path)
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                all_text.append(text.strip())

    full_text = "\n\n".join(all_text).strip()
    if not full_text:
        return {"status": "error", "message": "No extractable text found in PDFs."}

    chunks = split_text(full_text)
    if not chunks:
        return {"status": "error", "message": "No usable text found in PDFs."}

    added = ingest_text(chunks, replace=replace)
    return {"status": "ok", "chunks_added": added}


@app.post("/reset")
def reset():
    reset_store()
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
