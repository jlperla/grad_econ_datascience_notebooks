import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from openai import OpenAI
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from handwriting_ocr import MODEL, OCR_PROMPT, PAGE_SCALE, plain_text, transcribe

DATA = ROOT / "slides/data/handwriting"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["development", "test"], default="development")
    parser.add_argument("--mode", choices=["pages", "rows"], default="pages")
    parser.add_argument("--output", type=Path, default=ROOT / "build/handwriting/chandra")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pages = pd.read_csv(DATA / "pages.csv")
    boxes = pd.read_csv(DATA / "labels.csv", usecols=[
        "row_id", "page_id", "split", "x", "y", "width", "height",
    ])
    client = OpenAI(
        base_url=os.environ.get("OCR_BASE_URL", "http://127.0.0.1:8081/v1"),
        api_key=os.environ.get("OCR_API_KEY", "local"), timeout=300,
    )
    model = os.environ.get("OCR_MODEL", MODEL)
    records, responses = [], []
    for page in pages.loc[pages.split.eq(args.split)].itertuples():
        image = Image.open(DATA / page.image)
        images = [(page.page_id, image.resize(
            (image.width * PAGE_SCALE, image.height * PAGE_SCALE),
            Image.Resampling.LANCZOS,
        ))]
        if args.mode == "rows":
            images = [
                (row.row_id, image.crop((row.x, row.y,
                                        row.x + row.width, row.y + row.height))
                 .resize((round(row.width * 128 / row.height), 128),
                         Image.Resampling.LANCZOS))
                for row in boxes.loc[boxes.page_id.eq(page.page_id)].itertuples()
            ]
        for item_id, item in images:
            response = transcribe(item, client, model,
                                  8192 if args.mode == "pages" else 256)
            html = response.choices[0].message.content
            if args.mode == "pages":
                (args.output / f"{item_id}.html").write_text(html)
            records.append({
                "item_id": item_id, "page_id": page.page_id, "split": args.split,
                "finish_reason": response.choices[0].finish_reason,
                "text": plain_text(html),
                "completion_tokens": response.usage.completion_tokens,
            })
            responses.append({"item_id": item_id, "response": response.model_dump()})
            print(item_id, response.choices[0].finish_reason, flush=True)
    stem = f"{args.split}_{args.mode}"
    pd.DataFrame(records).to_csv(args.output / f"{stem}.csv", index=False)
    (args.output / f"{stem}_responses.jsonl").write_text(
        "\n".join(json.dumps(item) for item in responses) + "\n"
    )
    (args.output / f"{stem}_run.json").write_text(json.dumps({
        "model": model, "model_revision": os.environ.get("OCR_REVISION"),
        "date_utc": datetime.now(UTC).isoformat(),
        "temperature": 0, "max_tokens": 8192 if args.mode == "pages" else 256,
        "prompt": OCR_PROMPT,
        "prompt_sha256": hashlib.sha256(OCR_PROMPT.encode()).hexdigest(),
        "input": (f"Page crop resized by {PAGE_SCALE} with LANCZOS" if args.mode == "pages"
                  else "ALTO row crop resized to height 128 with LANCZOS"),
        "packages": {name: importlib.metadata.version(name)
                     for name in ["chandra-ocr", "openai", "Pillow"]},
        "runtime": os.environ.get("OCR_RUNTIME", "OpenAI-compatible server"),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
