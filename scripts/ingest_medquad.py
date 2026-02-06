import argparse
import os
import sys
from typing import Iterable

from datasets import load_dataset
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from src.helper import build_index, dedupe_pairs


def _get_field(row: dict, keys: Iterable[str]) -> str:
    for key in keys:
        if key in row and row[key]:
            return str(row[key]).strip()
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index for MedQuAD Q/A pairs.")
    parser.add_argument("--dataset", default="keivalya/MedQuad-MedicalQnADataset")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=0, help="Limit rows for quick tests")
    args = parser.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    if args.limit and args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    pairs = []
    for row in ds:
        question = _get_field(row, ["Question", "question", "QUESTION"])
        answer = _get_field(row, ["Answer", "answer", "ANSWER"])
        if not question or not answer:
            continue
        pairs.append((question, answer))

    pairs = dedupe_pairs(pairs)
    total = build_index(pairs)
    print(f"Indexed {total} Q/A pairs from {args.dataset}:{args.split}")


if __name__ == "__main__":
    main()
