import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

CHUNKS_PATH = Path("data/chunks.json")
INDEX_PATH = Path("faiss.index")
CHUNKS_NPY_PATH = Path("chunks.npy")

def main():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError("data/chunks.json not found")
    
    with open(CHUNKS_PATH, "r") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} text chunks")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(
        chunks,
        show_progress_bar = True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    print("Embeddings shape: ", embeddings.shape)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"FAISS index contains {index.ntotal} vectors")

    faiss.write_index(index, str(INDEX_PATH))
    np.save(CHUNKS_NPY_PATH, np.array(chunks, dtype = object))

    print("FAISS index saved to faiss.index")
    print("Chunks saved to chunks.npy")

if __name__ == "__main__":
    main()
