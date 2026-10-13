import argparse
import base64
import os
from io import BytesIO, StringIO
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from chandra.prompts import OCR_PROMPT
from openai import OpenAI
from PIL import Image

FIELDS = [
    "surname", "given_names", "birth_year", "birthplace", "nationality",
    "marital_status", "relationship", "education", "occupation", "employer",
]
MODEL = "datalab-to/chandra-ocr-2"
PAGE_SCALE = 2


def transcribe(image, client, model, max_tokens):
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{encoded}"}},
            {"type": "text", "text": OCR_PROMPT},
        ]}],
        temperature=0,
        max_tokens=max_tokens,
    )


def read_table(html):
    return pd.read_html(
        StringIO(html), keep_default_na=False,
        converters={i: str for i in range(10)},
    )[0]


def plain_text(html):
    return " ".join(BeautifulSoup(html, "html.parser").get_text(" ").split())


def main():
    parser = argparse.ArgumentParser()
    sample = (Path(__file__).resolve().parents[1] / "slides/data/handwriting/pages"
              / "AD075DP_D2M8_275_0025_left_page.png")
    parser.add_argument("image", type=Path, nargs="?", default=sample)
    parser.add_argument("--output", type=Path, default=Path("build/handwriting/page"))
    args = parser.parse_args()
    client = OpenAI(
        base_url=os.environ.get("OCR_BASE_URL", "http://127.0.0.1:8081/v1"),
        api_key=os.environ.get("OCR_API_KEY", "local"), timeout=300,
    )
    image = Image.open(args.image)
    image = image.resize((image.width * PAGE_SCALE, image.height * PAGE_SCALE),
                         Image.Resampling.LANCZOS)
    response = transcribe(image, client, os.environ.get("OCR_MODEL", MODEL), 8192)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(response.model_dump_json(indent=2))
    html = response.choices[0].message.content
    args.output.with_suffix(".html").write_text(html)
    table = read_table(html)
    table.columns = FIELDS
    table.to_csv(args.output.with_suffix(".csv"), index=False)
    print(table.to_string(index=False))
    print("finish_reason:", response.choices[0].finish_reason)


if __name__ == "__main__":
    main()
