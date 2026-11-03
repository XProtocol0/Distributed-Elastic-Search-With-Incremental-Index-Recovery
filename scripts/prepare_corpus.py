"""Download Simple English Wikipedia and cut it down to an abstracts corpus (D-026).

    python scripts/prepare_corpus.py [--limit 100000] [--out data/corpus.jsonl]

Wikimedia no longer publishes abstract dumps, so this takes the Hugging Face
`wikimedia/wikipedia` parquet file and keeps each article's lead paragraph.
Output is JSON lines: {"doc_id", "title", "text"}. Everything under data/ is
gitignored. Needs pyarrow (`pip install -e ".[data]"`).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

DEFAULT_URL = (
    "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/"
    "refs%2Fconvert%2Fparquet/20231101.simple/train/0000.parquet"
)
MIN_WORDS = 8  # skip stubs whose lead paragraph is just a title or a list header


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default=DEFAULT_URL, help="parquet file with id/title/text columns")
    parser.add_argument("--raw", type=Path, default=Path("data/raw/simplewiki-20231101.parquet"))
    parser.add_argument("--out", type=Path, default=Path("data/corpus.jsonl"))
    parser.add_argument("--limit", type=int, default=100_000)
    return parser.parse_args(argv)


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"using cached {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url} -> {dest}")
    with urllib.request.urlopen(url) as response, open(partial, "wb") as f:
        shutil.copyfileobj(response, f, length=1 << 20)
    partial.rename(dest)


def lead_paragraph(text: str) -> str:
    """The first non-empty paragraph of an article."""
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if paragraph:
            return paragraph
    return ""


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print('pyarrow is required: pip install -e ".[data]"', file=sys.stderr)
        return 2

    download(args.url, args.raw)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for batch in pq.ParquetFile(args.raw).iter_batches(columns=["id", "title", "text"]):
            for row in batch.to_pylist():
                abstract = lead_paragraph(row["text"])
                if len(abstract.split()) < MIN_WORDS:
                    continue
                record = {"doc_id": f"wiki-{row['id']}", "title": row["title"], "text": abstract}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                if written >= args.limit:
                    break
            if written >= args.limit:
                break
    print(f"wrote {written} abstracts to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
