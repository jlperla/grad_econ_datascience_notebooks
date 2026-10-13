import argparse
import base64
import hashlib
import json
import os
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import pandas as pd
from build_handwriting_argentina_holdout import download_image
from chandra.prompts import OCR_PROMPT
from openai import OpenAI
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "build/handwriting/argentina"
REPOSITORY = "Saucy-Stream/Argentina_1895_census_HTR_code"
REVISION = "1048ba9c12863283ad17b6cde11684e0ed42885d"
FILENAME = "Cordoba_Calamuchita_Cañada de Álvarez_image 10 of 168.jpg"
SOURCE_URL = (
    f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/test/"
    + quote(FILENAME)
)
MODEL = "Saucystream/fine_tuned_chandra"
MODEL_REVISION = "d94b4ca422dd87e06b55abdd955bb199879a545a"
BASE_MODEL = "datalab-to/chandra"
BASE_REVISION = "bd40c21576564d31ba99f204e28ccd5c4c1751fa"
CHECKPOINTS = {
    "finetuned": (MODEL, MODEL_REVISION, "build/handwriting/argentina-model"),
    "blur": ("Saucystream/fine_tuned_chandra_blur",
             "6a103e674149b2aed54e824fe41ab09838443e7e",
             "build/handwriting/argentina-blur-model"),
    "chandra1": (BASE_MODEL, BASE_REVISION, BASE_MODEL),
    "chandra2": ("datalab-to/chandra-ocr-2",
                 "af93b47dba1b47b6640c86ccf487ed2260ab9a09",
                 "datalab-to/chandra-ocr-2"),
}

# The prompt and example format are from the author's CensusDataset.py.
EXAMPLE = (
    '{"headers":["marginalia","num","A","B","C","D","E","F",'
    '"H","I","J","K","L","M","N","O","P"],\n"data": \n[{\n'
    '\t"marginalia": "*",\n\t"num": 1,\n\t"A": ["Ovideo","Carlos"],\n'
    '\t"B": "V",\n\t"C": 56,\n\t"D": "V",\n\t"E": "Argentina",\n'
    '\t"F": "Córdoba",\n\t"H": "Estanciero",\n\t"I": "si",\n'
    '\t"J": "*",\n\t"K": "*",\n\t"L": ["*","*"],\n\t"M": "*",\n'
    '\t"N": "*",\n\t"O": ["*","*"],\n\t"P": "*"\n},\n{...'
)
SYSTEM_PROMPT = (
    "You are an expert in 1800s Argentinian handwriting who will transcribe "
    "an Argentinian census document into a JSON file. The document consists "
    "of a scanned table with machine typed headers and hand written data. "
    "Make use of the information in the headers to find a reasonable "
    "transcription for each field. Some fields only accept elements such "
    "as 'si' or 'no', or 'V' or 'M'. Empty fields should be transcribed as "
    "an asterisk (*). Split columns should be transcribed as a list. Here "
    f"is a made up example output to illustrate the format: '{EXAMPLE}'"
)
USER_PROMPT = (
    "Transcribe this document into a JSON file. Only output the table, not "
    "any other explanatory text. Do not fill the empty fields with filler "
    "words like 'si' or 'no'. Omit completely empty rows. Make sure to "
    "transcribe any potential note written in the margin of the table to "
    "the correct row. Make sure that your transcription is reasonable "
    "and accurate."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=CHECKPOINTS, default="finetuned")
    parser.add_argument("--prompt", choices=["json", "html"], default="json")
    parser.add_argument("--page", choices=["training", "holdout"], default="training")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    cache = CACHE / "holdout" if args.page == "holdout" else CACHE
    output = args.output or cache / f"{args.checkpoint}-{args.prompt}"
    model, revision, default_request_model = CHECKPOINTS[args.checkpoint]
    if args.page == "holdout":
        image_bytes, page_source = download_image()
    else:
        CACHE.mkdir(parents=True, exist_ok=True)
        with urlopen(SOURCE_URL) as response:
            image_bytes = response.read()
        (CACHE / "source.jpg").write_bytes(image_bytes)
        page_source = {
            "source": "Argentina 1895 census, Córdoba, Calamuchita, Cañada de Álvarez",
            "source_url": SOURCE_URL, "filename": FILENAME,
            "split": "author training set",
            "split_source": f"https://github.com/{REPOSITORY}/blob/{REVISION}/slurm-outputs/slurm-6653862.out",
            "evaluation": "Training-page comparison; not unseen-page accuracy",
        }
    print(f"Downloaded {page_source['source_url']}\nTo {cache}", flush=True)
    if args.download_only:
        return

    image = Image.open(BytesIO(image_bytes))
    original_size = image.size
    image.thumbnail((2048, 1024))
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    request_model = os.environ.get("OCR_MODEL", default_request_model)
    client = OpenAI(
        base_url=os.environ.get("OCR_BASE_URL", "http://127.0.0.1:8084/v1"),
        api_key=os.environ.get("OCR_API_KEY", "local"), timeout=600,
    )
    user_prompt = USER_PROMPT if args.prompt == "json" else OCR_PROMPT
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {
            "url": f"data:image/png;base64,{encoded}"}},
        {"type": "text", "text": user_prompt},
    ]}]
    if args.prompt == "json":
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    response = client.chat.completions.create(
        model=request_model, temperature=0, max_tokens=8192, messages=messages,
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "response.json").write_text(response.model_dump_json(indent=2))
    (output / "source.json").write_text(json.dumps({
        **page_source, "repository": REPOSITORY,
        "repository_revision": REVISION,
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "original_size": original_size, "input_size": image.size,
        "input_resize": "Pillow thumbnail, fit within 2048 x 1024, BICUBIC",
        "input_sha256": hashlib.sha256(buffer.getvalue()).hexdigest(),
        "image_distribution": "Downloaded on demand; no source scan bundled",
        "model": model, "model_revision": revision,
        "base_model": BASE_MODEL if args.checkpoint in ["finetuned", "blur"] else model,
        "base_revision": BASE_REVISION if args.checkpoint in ["finetuned", "blur"] else revision,
        "request_model": request_model,
        "system_prompt": SYSTEM_PROMPT if args.prompt == "json" else None,
        "user_prompt": user_prompt, "prompt": args.prompt,
        "prompt_source": (
            f"https://github.com/{REPOSITORY}/blob/{REVISION}/CensusDataset.py"
            if args.prompt == "json" else
            "https://github.com/datalab-to/chandra/blob/v0.2.0/chandra/prompts.py"
        ),
        "prompt_function": "json_training_transform" if args.prompt == "json" else "OCR_PROMPT",
        "temperature": 0, "max_tokens": 8192,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "finish_reason": response.choices[0].finish_reason,
        "runtime": os.environ.get("OCR_RUNTIME", "OpenAI-compatible server"),
        "date_utc": datetime.now(UTC).isoformat(),
    }, ensure_ascii=False, indent=2) + "\n")
    if args.prompt == "html":
        (output / "prediction.html").write_text(response.choices[0].message.content)
        print("finish_reason:", response.choices[0].finish_reason)
        return
    prediction = json.loads(response.choices[0].message.content)
    (output / "prediction.json").write_text(json.dumps(
        prediction, ensure_ascii=False, indent=2) + "\n")
    people = pd.DataFrame(prediction["data"])
    if args.checkpoint in ["finetuned", "blur"]:
        names = pd.DataFrame(people["A"].tolist(), columns=["surname", "given_names"])
    else:
        names = people["A"].map(
            lambda value: " ".join(value) if isinstance(value, list) else value
        ).to_frame("full_name")
    people = pd.concat([names, people[["B", "C", "H", "I", "K"]]], axis=1)
    people = people.rename(columns={
        "B": "sex", "C": "age", "H": "occupation",
        "I": "literacy", "K": "property_owner",
    })
    people.to_csv(output / "prediction.csv", index=False)
    print(people.to_string(index=False))
    print("finish_reason:", response.choices[0].finish_reason)


if __name__ == "__main__":
    main()
