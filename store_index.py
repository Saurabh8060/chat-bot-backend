"""Optional script to ingest text from a local file into Pinecone."""
import os
from dotenv import load_dotenv

from src.helper import ensure_pinecone_index, split_text, ingest_text

load_dotenv()

FILE_PATH = os.getenv("INGEST_FILE", "./data.txt")
REPLACE = os.getenv("REPLACE", "true").lower() == "true"


def main():
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(FILE_PATH)
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    ensure_pinecone_index()
    chunks = split_text(text)
    added = ingest_text(chunks, replace=REPLACE)
    print(f"Ingested {added} chunks from {FILE_PATH}")


if __name__ == "__main__":
    main()
