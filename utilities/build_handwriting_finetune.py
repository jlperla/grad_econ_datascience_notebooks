import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from chandra.prompts import OCR_PROMPT
from openai import OpenAI
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from handwriting_ocr import transcribe

DEST = ROOT / "slides/data/handwriting/appen"
DATASET = "fwgpiyawudk/appen-thai-doc"
DATA_REVISION = "4d00648ef335a1951acd5fd85ffee9be813bdf77"
MODEL = "fwgpiyawudk/ChandraOCR2_AppenDoc-merged"
MODEL_REVISION = "dde20ac03d4dbc851af082576c7b34f095cf12e0"
FILENAME = "THAI_RECEIPT_003149.jpg"
SOURCE = f"https://huggingface.co/datasets/{DATASET}/resolve/{DATA_REVISION}/train"


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    with urlopen(f"{SOURCE}/{FILENAME}") as response:
        image_bytes = response.read()
    (DEST / FILENAME).write_bytes(image_bytes)
    client = OpenAI(
        base_url=os.environ.get("OCR_BASE_URL", "http://127.0.0.1:8083/v1"),
        api_key=os.environ.get("OCR_API_KEY", "local"), timeout=300,
    )
    request_model = os.environ.get("OCR_MODEL", "build/handwriting/appen-model")
    image = Image.open(DEST / FILENAME)
    w, h = image.size
    crop_box = (int(w * .10), int(h * .32), int(w * .90), int(h * .415))
    image = image.crop(crop_box)
    image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    image.save(DEST / "input.png")
    response = transcribe(image, client, request_model, 2048)
    (DEST / "response.json").write_text(response.model_dump_json(indent=2))
    html = response.choices[0].message.content
    (DEST / "prediction.html").write_text(html)
    table = pd.read_html(StringIO(html), keep_default_na=False,
                         converters={i: str for i in range(4)})[0]
    table.columns = ["quantity", "description", "unit_price", "amount"]
    table.to_csv(DEST / "prediction.csv", index=False)
    (DEST / "prediction.json").write_text(json.dumps(
        table.to_dict(orient="records"), ensure_ascii=False, indent=2) + "\n")
    (DEST / "source.json").write_text(json.dumps({
        "source": "Appen Limited, OCR image data for Thai documents",
        "source_url": "https://www.kaggle.com/datasets/appenlimited/ocr-image-data-for-thai-documents",
        "license": "CC-BY-SA-4.0", "dataset": DATASET,
        "dataset_revision": DATA_REVISION, "filename": FILENAME,
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "original_image_changes": "None", "model": MODEL,
        "input_crop_box": crop_box, "input_size": image.size,
        "input_resize": "Fit within 1600 x 1600, LANCZOS",
        "input_sha256": hashlib.sha256((DEST / "input.png").read_bytes()).hexdigest(),
        "model_revision": MODEL_REVISION, "request_model": request_model,
        "date_utc": datetime.now(UTC).isoformat(),
        "prompt": OCR_PROMPT, "temperature": 0, "max_tokens": 2048,
        "finish_reason": response.choices[0].finish_reason,
        "runtime": os.environ.get("OCR_RUNTIME", "OpenAI-compatible server"),
        "evaluation": "Published training-split image; inference demonstration only",
    }, ensure_ascii=False, indent=2) + "\n")
    (DEST / "ATTRIBUTION.txt").write_text(
        "Appen Limited, OCR image data for Thai documents.\n"
        "Source: https://www.kaggle.com/datasets/appenlimited/ocr-image-data-for-thai-documents\n"
        "Image: THAI_RECEIPT_003149.jpg, unchanged.\n"
        "License: CC BY-SA 4.0, https://creativecommons.org/licenses/by-sa/4.0/\n"
        "Hugging Face image mirror: fwgpiyawudk/appen-thai-doc.\n"
        "input.png: crop to the printed item header and first row, resized to width 1600.\n"
        "Prediction HTML, CSV, and derived JSON are model outputs.\n"
        "This example and its adapted outputs are distributed under CC BY-SA 4.0.\n"
    )
    print(html)


if __name__ == "__main__":
    main()
