import argparse
import base64
import hashlib
import importlib.metadata
import json
import os
import re
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["line", "name", "age", "occupation", "earnings"]
QWEN_PROMPT = (
    'Extract every row of this census table as JSON. Return only an object '
    'with key "rows", a list of objects with string fields "line", "name", '
    '"age", "occupation", "earnings". Read only the image. Preserve spelling '
    'and abbreviations, including ditto marks. Do not fill blanks or infer '
    'missing values. Use an empty string for an empty cell. Keep every row in '
    'printed order. No commentary.'
)
MODELS = {"chandra": "datalab-to/chandra-ocr-2",
          "qwen27b": "qwen3.8:27b-mxfp8", "qwen2b": "qwen3.5:2b"}


def parse_rows(text, workflow):
    if workflow == "chandra":
        soup = BeautifulSoup(text, "html.parser")
        rows = []
        for table in soup.find_all("table"):
            headers = [cell.get_text(" ", strip=True).casefold()
                       for cell in table.find_all("th")
                       for repeat in range(int(cell.get("colspan", 1)))]
            for tr in table.find_all("tr"):
                cells = tr.find_all("td")
                if not cells:
                    continue
                values = [cell.get_text(" ", strip=True)
                          if int(cell.get("colspan", 1)) == 1
                          and int(cell.get("rowspan", 1)) == 1 else float("nan")
                          for cell in cells
                          for repeat in range(int(cell.get("colspan", 1)))]
                row = dict.fromkeys(FIELDS, float("nan"))
                if len(values) == len(headers):
                    row.update({field: values[headers.index(field)] for field in FIELDS
                                if headers.count(field) == 1})
                rows.append(row)
        return pd.DataFrame(rows, columns=FIELDS)
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        rows = json.loads(text)["rows"]
    except json.JSONDecodeError:
        start = re.search(r'"rows"\s*:\s*\[', text)
        remaining = text[start.end():].lstrip() if start else ""
        rows = []
        while remaining.startswith("{"):
            try:
                row, end = json.JSONDecoder().raw_decode(remaining)
            except json.JSONDecodeError:
                break
            rows.append(row)
            remaining = remaining[end:].lstrip().removeprefix(",").lstrip()
    return pd.DataFrame(rows, columns=FIELDS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", choices=list(MODELS), default="chandra")
    parser.add_argument("--split", choices=["development", "test"], default="development")
    parser.add_argument("--data", type=Path, default=ROOT / "build/handwriting/toronto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reparse", action="store_true")
    args = parser.parse_args()
    output = args.output or args.data / args.workflow
    output.mkdir(parents=True, exist_ok=True)
    panels = pd.read_csv(args.data / "panels.csv")
    if not args.reparse:
        run_model(args, output, panels)
    responses_path = output / f"{args.split}_responses.jsonl"
    records, structure = [], []
    panel_info = panels.set_index("panel_id")
    for item in map(json.loads, responses_path.read_text().splitlines()):
        panel = panel_info.loc[item["panel_id"]]
        payload = item["response"]
        if args.workflow == "chandra":
            text = payload["choices"][0]["message"]["content"] or ""
            finish = payload["choices"][0]["finish_reason"]
        else:
            text = payload["message"]["content"]
            finish = payload["done_reason"]
        rows = parse_rows(text, args.workflow)
        rows_emitted = len(rows)
        rows["line"] = rows.line.astype("string").str.strip()
        expected = [str(n) for n in range(panel.first_line, panel.last_line + 1)]
        rows = rows.loc[~rows.line.duplicated(keep=False) & rows.line.isin(expected)]
        rows = rows.set_index("line").reindex(expected)
        structure.append({
            "panel_id": item["panel_id"], "page_id": panel.page_id,
            "split": panel.split, "rows_emitted": rows_emitted,
            "finish_reason": finish,
            **{f"{field}_available": int(rows[field].notna().sum())
               for field in FIELDS[1:]},
            "unavailable_columns": ",".join(
                field for field in FIELDS[1:] if rows[field].isna().all()),
        })
        for row in rows.reset_index().to_dict("records"):
            records.append({"page_id": panel.page_id, "panel_id": item["panel_id"],
                            "split": panel.split, **row})
        print(item["panel_id"], finish,
              f"age={rows.age.notna().sum()}/10",
              f"earnings={rows.earnings.notna().sum()}/10", flush=True)
    pd.DataFrame(records).to_csv(output / f"{args.split}_predictions.csv",
                                 index=False, na_rep="NaN")
    pd.DataFrame(structure).to_csv(output / f"{args.split}_structure.csv", index=False)


def run_model(args, output, panels):
    model = os.environ.get("OCR_MODEL", MODELS[args.workflow])
    prompt = QWEN_PROMPT
    if args.workflow == "chandra":
        from chandra.prompts import OCR_PROMPT
        prompt = OCR_PROMPT
    client = OpenAI(base_url=os.environ.get("OCR_BASE_URL", "http://127.0.0.1:8081/v1"),
                    api_key=os.environ.get("OCR_API_KEY", "local"), timeout=600)
    responses_path = output / f"{args.split}_responses.jsonl"
    with responses_path.open("w") as saved:
        for panel in panels.loc[panels.split.eq(args.split)].itertuples():
            encoded = base64.b64encode((args.data / panel.image).read_bytes()).decode()
            if args.workflow == "chandra":
                response = client.chat.completions.create(
                    model=model, temperature=0, max_tokens=4096,
                    messages=[{"role": "user", "content": [
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{encoded}"}},
                        {"type": "text", "text": prompt},
                    ]}],
                )
                payload = response.model_dump()
                text = response.choices[0].message.content or ""
            else:
                request = urllib.request.Request(
                    os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434") + "/api/chat",
                    data=json.dumps({
                        "model": model, "stream": False, "think": False,
                        "messages": [{"role": "user", "content": prompt,
                                      "images": [encoded]}],
                        "options": {"temperature": 0, "num_predict": 4096,
                                    "num_ctx": 16384}, "keep_alive": "10m",
                    }).encode(), headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=600) as response:
                    payload = json.load(response)
                text = payload["message"]["content"]
            saved.write(json.dumps({"panel_id": panel.panel_id, "response": payload}) + "\n")
            saved.flush()
            (output / f"{panel.panel_id}.txt").write_text(text)
            print(panel.panel_id, "response saved", flush=True)
    revision, runtime = os.environ.get("OCR_REVISION"), os.environ.get("OCR_RUNTIME")
    model_details = None
    if args.workflow != "chandra":
        host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        with urllib.request.urlopen(host + "/api/tags") as response:
            installed = json.load(response)["models"]
        model_details = next(item for item in installed if item["name"] == model)
        revision = model_details["digest"]
        with urllib.request.urlopen(host + "/api/version") as response:
            runtime = "Ollama " + json.load(response)["version"]
    (output / f"{args.split}_run.json").write_text(json.dumps({
        "model": model, "model_revision": revision, "runtime": runtime,
        "model_details": model_details, "date_utc": datetime.now(UTC).isoformat(),
        "workflow": args.workflow, "temperature": 0, "max_tokens": 4096,
        "thinking": False if args.workflow != "chandra" else None,
        "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "inputs": panels.loc[panels.split.eq(args.split)].to_dict("records"),
        "packages": {name: importlib.metadata.version(name)
                     for name in ["openai", "pandas", "Pillow"]},
        "input": "Identical annotated cell panels; printed row order; no reference values",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
