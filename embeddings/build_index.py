import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("data/chunks.json")
EMBEDDINGS_PATH = Path("embeddings.npy")
CHUNKS_NPY_PATH = Path("chunks.npy")


def main():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError("data/chunks.json not found")

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} text chunks")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        chunks,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    print("Embeddings shape:", embeddings.shape)

    np.save(EMBEDDINGS_PATH, embeddings)
    np.save(CHUNKS_NPY_PATH, np.array(chunks, dtype=object))

    print(f"Embeddings saved to {EMBEDDINGS_PATH}")
    print(f"Chunks saved to {CHUNKS_NPY_PATH}")


if __name__ == "__main__":
    main()
