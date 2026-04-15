import os
from dotenv import load_dotenv

# Avoid slow Pydantic plugin auto-discovery during app startup.
os.environ.setdefault("PYDANTIC_DISABLE_PLUGINS", "__all__")

# Load .env before importing helper
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.helper import answer_question

app = FastAPI(title="RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str | None = None
    question: str | None = None
    last_topic: str | None = None
    previous_user_message: str | None = None
    previous_assistant_message: str | None = None
    debug: bool | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(payload: ChatRequest):
    question = (payload.message or payload.question or "").strip()
    if not question:
        return {"answer": "Please provide a question.", "sources": []}

    last_topic = (payload.last_topic or "").strip() or None
    previous_user_message = (payload.previous_user_message or "").strip() or None
    previous_assistant_message = (payload.previous_assistant_message or "").strip() or None
    return answer_question(
        question,
        last_topic=last_topic,
        previous_user_message=previous_user_message,
        previous_assistant_message=previous_assistant_message,
        debug=bool(payload.debug),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
