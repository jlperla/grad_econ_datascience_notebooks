import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import pandas as pd
from bs4 import BeautifulSoup
from build_handwriting_argentina import CACHE, FILENAME, REPOSITORY, REVISION
from build_handwriting_argentina_holdout import (
    THESIS_URL,
    download_image,
    reference_rows,
)
from build_handwriting_scores import cell_metrics, cell_score, html_rows, html_value

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "slides/data/handwriting/argentina"
REFERENCE_URL = (
    f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/"
    "test_transcriptions/" + quote(Path(FILENAME).with_suffix(".txt").name)
)
HTML_COLUMNS = {
    "full_name": [1, 2], "sex": [3], "age": [4],
    "occupation": [8], "literacy": [9], "property_owner": [11],
}


def normalize(value):
    return " ".join(unicodedata.normalize("NFC", str(value)).split()) or "*"


def accent_fold(value):
    return "".join(c for c in unicodedata.normalize("NFD", normalize(value))
                   if not unicodedata.combining(c))


def json_rows(content):
    try:
        return json.loads(content)["data"]
    except json.JSONDecodeError:
        start = re.search(r'"data"\s*:\s*\[', content)
        if start is None:
            return []
        remaining = content[start.end():]
        rows = []
        while remaining.strip().startswith("{"):
            try:
                row, end = json.JSONDecoder().raw_decode(remaining.lstrip())
            except json.JSONDecodeError:
                break
            rows.append(row)
            remaining = remaining.lstrip()[end:].lstrip().removeprefix(",")
        return rows


def unique_rows(records):
    counts = Counter(number for number, _ in records if number is not None)
    return {number: row for number, row in records
            if number is not None and counts[number] == 1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--page", choices=["training", "holdout"], default="training")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--automatic-only", action="store_true")
    parser.add_argument("--reject-field", action="append", choices=HTML_COLUMNS, default=[])
    args = parser.parse_args()
    data = DATA / "holdout" if args.page == "holdout" else DATA
    cache = CACHE / "holdout" if args.page == "holdout" else CACHE
    predictions_dir = args.predictions or data
    output = args.output or cache / "scores"
    protocol = json.loads((data / "comparison_protocol.json").read_text())
    audits = json.loads((DATA.parent / "layout_audit.json").read_text())["argentina"]
    fields = protocol["fields"]
    if args.page == "holdout":
        download_image()
        reference = reference_rows()
        reference_bytes = (cache / "reference.csv").read_bytes()
        reference_url = THESIS_URL + "#page=92"
    else:
        CACHE.mkdir(parents=True, exist_ok=True)
        with urlopen(REFERENCE_URL) as response:
            reference_bytes = response.read()
        (CACHE / "reference.txt").write_bytes(reference_bytes)
        reference = list(csv.reader(reference_bytes.decode().splitlines()[3:]))
        reference_url = REFERENCE_URL
    truth = {
        int(row[1]): {field: normalize(" ".join(row[c] for c in cols)) for field, cols in
                      zip(fields, protocol["reference_columns"])}
        for row in reference
    }
    if args.page == "holdout":
        runs = {
            "Chandra 1 / HTML": predictions_dir / "chandra1-html",
            "Chandra 1 + LoRA / JSON": predictions_dir / "finetuned-json",
            "Chandra OCR 2 / HTML": predictions_dir / "chandra2-html",
            "Chandra 1 + blur LoRA / JSON": predictions_dir / "blur-json",
        }
    else:
        runs = {
            "Chandra 1 / JSON": predictions_dir / ("chandra1-json" if args.predictions else "chandra1"),
            "Chandra 1 + LoRA / JSON": predictions_dir / "finetuned-json" if args.predictions else predictions_dir,
            "Chandra OCR 2 / JSON": predictions_dir / ("chandra2-json" if args.predictions else "chandra2"),
            "Chandra 1 / HTML": predictions_dir / "chandra1-html",
            "Chandra OCR 2 / HTML": predictions_dir / "chandra2-html",
        }
    summaries, cells, sources, predictions, automatic_predictions = [], [], {}, [], []
    for model, path in runs.items():
        response_bytes = (path / "response.json").read_bytes()
        response = json.loads(response_bytes)
        source = json.loads((path / "source.json").read_text())
        response_hash = hashlib.sha256(response_bytes).hexdigest()
        sources[model] = {
            "model": source["model"], "revision": source["model_revision"],
            "input_sha256": source["input_sha256"], "response_sha256": response_hash,
        }
        summary = {"model": model,
                   "finish_reason": response["choices"][0]["finish_reason"]}
        content = response["choices"][0]["message"]["content"]
        audit = {} if args.automatic_only else audits.get(response_hash, {})
        if audit and (audit["reference_sha256"] != hashlib.sha256(reference_bytes).hexdigest()
                      or audit["input_sha256"] != source["input_sha256"]):
            print(f"{model}: recorded audit does not match source hashes; automatic alignment only")
            audit = {}
        records, automatic_records, assisted_numbers = [], [], set()
        if source.get("prompt") == "html":
            rows = html_rows(content)
            summary["tables"] = len(BeautifulSoup(content, "html.parser").find_all("table"))
            summary["table_columns"] = max((len(row) for row in rows), default=0)
            summary["format"] = "HTML"
            overrides = {row["output_row"]: row for row in audit.get("row_overrides", [])}
            summary["names_split"] = 0
            for i, row in enumerate(rows):
                number = html_value(row, [0])
                number = int(number) if number is not None and number.isdigit() else None
                values = {field: html_value(row, columns)
                          for field, columns in HTML_COLUMNS.items()}
                automatic_records.append((number, values))
                mapping = overrides.get(i)
                if mapping:
                    number = int(html_value(row, [mapping["number_column"]]))
                    values = {field: html_value(row, columns)
                              for field, columns in mapping["columns"].items()}
                    assisted_numbers.add(number)
                columns = (mapping["columns"] if mapping else HTML_COLUMNS)["full_name"]
                parts = [html_value(row, [c]) for c in columns]
                summary["names_split"] += int(len(parts) == 2 and all(parts))
                records.append((number, values))
            sources[model]["html_columns"] = HTML_COLUMNS
        else:
            rows = json_rows(content)
            summary["format"] = "JSON" if rows else "no complete records"
            summary["names_split"] = sum(
                isinstance(row.get("A"), list) and len(row["A"]) == 2 for row in rows)
            for row in rows:
                number = row.get("num", row.get("Número de orden"))
                number = int(number) if str(number).isdigit() else None
                name = row.get("A")
                values = dict(zip(fields, [
                    " ".join(name) if isinstance(name, list) else name,
                    row.get("B"), row.get("C"), row.get("H"), row.get("I"), row.get("K"),
                ]))
                records.append((number, values))
            automatic_records = records
        predicted, automatic = unique_rows(records), unique_rows(automatic_records)
        numbers = [number for number, _ in records if number is not None]
        for frame in [predicted, automatic]:
            for row in frame.values():
                for field in args.reject_field:
                    row[field] = None
        for destination, frame in [(predictions, predicted), (automatic_predictions, automatic)]:
            destination.extend({"model": model, "row": number,
                                **{field: frame.get(number, {}).get(field) for field in fields}}
                               for number in sorted(set(truth) | set(frame)))
        summary.update({
            "emitted_rows": len(rows), "aligned_rows": len(set(predicted) & set(truth)),
            "automatic_aligned_rows": len(set(automatic) & set(truth)),
            "assisted_rows": len(assisted_numbers),
            "unmatched_reference_rows": len(set(truth) - set(predicted)),
            "extra_rows": len(set(predicted) - set(truth)),
            "duplicate_rows": len(numbers) - len(set(numbers)),
            "automatic_only": args.automatic_only,
            "alignment": "source-audited" if assisted_numbers else "automatic",
            "rejected_fields": ",".join(args.reject_field),
        })
        model_cells = []
        for number, row in truth.items():
            for field in fields:
                value = predicted.get(number, {}).get(field)
                value = normalize(value) if value is not None else None
                automatic_value = automatic.get(number, {}).get(field)
                automatic_value = normalize(automatic_value) if automatic_value is not None else None
                reference_text = "" if row[field] == "*" else row[field]
                score = cell_score(reference_text, "" if value == "*" else value, field == "age")
                automatic_score = cell_score(
                    reference_text, "" if automatic_value == "*" else automatic_value)
                disagrees = score["available"] and not score["correct"]
                model_cells.append({
                    "model": model, "row": number, "field": field, **score,
                    "automatic_available": automatic_score["available"],
                    "automatic_correct": automatic_score["correct"],
                    "automatic_character_errors": automatic_score["character_errors"],
                    "assisted": number in assisted_numbers and score["available"],
                    "binary_no_equivalent": (disagrees and field in ["literacy", "property_owner"]
                                             and row[field] in ["no", "nó"] and value in ["no", "nó"]),
                    "period_only": (disagrees and value.replace(".", "")
                                    == row[field].replace(".", "")),
                    "ditto_only": disagrees and row[field] == "''" and value == '"',
                    "accent_folded_disagrees": (score["available"]
                                               and accent_fold(value) != accent_fold(row[field])),
                    "uncertain_reference": (
                        [number, field] in protocol["uncertain_cells"]
                        if args.page == "holdout" else number == 14 and field == "occupation"),
                })
        table = pd.DataFrame(model_cells)
        available = table.loc[table.available]
        filled = available.loc[available.filled]
        unflagged = available.loc[~available.uncertain_reference]
        summary.update({
            **cell_metrics(table),
            "errors": int(available.correct.eq(0).sum()),
            "error_rate": 1 - available.correct.mean(),
            "unavailable_cells": int((~table.available).sum()),
            "filled_errors": int(filled.correct.eq(0).sum()),
            "filled_error_rate": 1 - filled.correct.mean(),
            "people": len(truth),
            "unflagged_available": len(unflagged),
            "unflagged_errors": int(unflagged.correct.eq(0).sum()),
            "unflagged_error_rate": 1 - unflagged.correct.mean(),
            "no_normalized_errors": int(available.correct.eq(0).sum()
                                        - available.binary_no_equivalent.sum()),
            "period_only_errors": int(available.period_only.sum()),
            "ditto_only_errors": int(available.ditto_only.sum()),
            "accent_folded_errors": int(available.accent_folded_disagrees.sum()),
        })
        summaries.append(summary)
        cells.extend(model_cells)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False, na_rep="NaN")
    pd.DataFrame(predictions).to_csv(output / "predictions.csv", index=False, na_rep="NaN")
    pd.DataFrame(automatic_predictions).to_csv(output / "automatic_predictions.csv",
                                             index=False, na_rep="NaN")
    field_scores = pd.DataFrame([
        {"model": model, "field": field, **cell_metrics(group)}
        for (model, field), group in pd.DataFrame(cells).groupby(["model", "field"], sort=False)
    ])
    field_scores.to_csv(output / "fields.csv", index=False, na_rep="NaN")
    (output / "layout_diagnostic.json").write_text(json.dumps([
        {"model": row["model"], "automatic_cells": row["automatic_cells"],
         "source_audited_cells": row["available_cells"],
         "assisted_cells": row["assisted_cells"], "disagreements": row["errors"],
         "status": "Source-audited alignment; automatic scores reported separately",
         "audit": "../../layout_audit.json" if args.page == "training"
         else "../../../layout_audit.json"}
        for row in summaries if row["assisted_cells"]
    ], indent=2) + "\n")
    (output / "source.json").write_text(json.dumps({
        "reference_url": reference_url,
        "reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "reference_distribution": "Downloaded on demand; not bundled",
        "protocol": protocol, "runs": sources,
        "layout_audit": "../../layout_audit.json" if args.page == "training" else "../../../layout_audit.json",
        "automatic_only": args.automatic_only,
    }, ensure_ascii=False, indent=2) + "\n")
    print(pd.DataFrame(summaries)[[
        "model", "automatic_cells", "available_cells", "assisted_cells",
        "errors", "error_rate", "character_accuracy", "digit_accuracy",
        "automatic_exact_yield", "exact_yield",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
