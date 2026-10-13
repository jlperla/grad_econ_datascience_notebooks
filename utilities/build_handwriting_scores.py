import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from handwriting_ocr import FIELDS

DATA = ROOT / "slides/data/handwriting"


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def edit_distance(reference, prediction):
    previous = list(range(len(prediction) + 1))
    for i, a in enumerate(reference, 1):
        current = [i]
        for j, b in enumerate(prediction, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def html_rows(html):
    rows = []
    pending = {}
    for fragment in re.findall(r"<tr\b[^>]*>.*?</tr>", html, re.I | re.S):
        entries = BeautifulSoup(fragment, "html.parser").find_all("td")
        if not entries:
            continue
        row = {column: cell for column, (cell, remaining) in pending.items()}
        pending = {column: (cell, remaining - 1)
                   for column, (cell, remaining) in pending.items() if remaining > 1}
        column = 0
        for entry in entries:
            while column in row:
                column += 1
            width = int(entry.get("colspan", 1))
            span = tuple(range(column, column + width))
            for br in entry.find_all("br"):
                br.replace_with(" ")
            height = int(entry.get("rowspan", 1))
            cell = (entry.get_text().strip(), span, height)
            row.update({c: cell for c in span})
            if height > 1:
                pending.update({c: (cell, height - 1) for c in span})
            column += width
        rows.append(row)
    return rows


def html_value(row, columns):
    parts = {}
    for column in columns:
        if column not in row:
            return None
        value, span, height = row[column]
        if height != 1 or not set(span).issubset(columns):
            return None
        parts[span] = value
    return " ".join(parts.values())


def cell_score(reference, prediction, numeric=False):
    available = prediction is not None
    filled = reference != ""
    characters = len(reference)
    errors = edit_distance(reference, prediction) if available and filled else 0
    digits = numeric and reference.isascii() and reference.isdigit()
    return {
        "available": available, "filled": filled,
        "correct": int(reference == prediction) if available else None,
        "characters": characters,
        "available_characters": characters if available else 0,
        "character_errors": errors,
        "digits": characters if digits else 0,
        "available_digits": characters if digits and available else 0,
        "digit_errors": errors if digits else 0,
    }


def cell_metrics(group):
    available = group.loc[group.available]
    filled = group.loc[group.filled]
    filled_available = filled.loc[filled.available]
    blank = group.loc[~group.filled]
    blank_available = blank.loc[blank.available]
    characters = int(group.available_characters.sum())
    digits = int(group.available_digits.sum())
    automatic_characters = int(group.loc[group.automatic_available, "characters"].sum())
    return {
        "cells": len(group), "available_cells": len(available),
        "coverage": group.available.mean(), "correct": int(available.correct.sum()),
        "accuracy": available.correct.mean(),
        "exact_yield": available.correct.sum() / len(group),
        "filled_cells": len(filled), "filled_available": len(filled_available),
        "filled_coverage": filled.available.mean(),
        "filled_accuracy": filled_available.correct.mean(),
        "blank_cells": len(blank), "blank_available": len(blank_available),
        "blank_accuracy": blank_available.correct.mean(),
        "false_fills": int(blank_available.correct.eq(0).sum()),
        "reference_characters": int(group.characters.sum()),
        "available_characters": characters,
        "character_coverage": characters / group.characters.sum()
        if group.characters.sum() else float("nan"),
        "character_errors": int(group.character_errors.sum()),
        "character_accuracy": 1 - group.character_errors.sum() / characters
        if characters else float("nan"),
        "reference_digits": int(group.digits.sum()), "available_digits": digits,
        "digit_coverage": digits / group.digits.sum()
        if group.digits.sum() else float("nan"),
        "digit_errors": int(group.digit_errors.sum()),
        "digit_accuracy": 1 - group.digit_errors.sum() / digits
        if digits else float("nan"),
        "automatic_cells": int(group.automatic_available.sum()),
        "automatic_coverage": group.automatic_available.mean(),
        "automatic_exact_yield": group.automatic_correct.sum() / len(group),
        "automatic_character_accuracy":
        1 - group.automatic_character_errors.sum() / automatic_characters
        if automatic_characters else float("nan"),
        "assisted_cells": int(group.assisted.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--split", choices=["development", "test"], default="test")
    parser.add_argument("--pages-only", action="store_true")
    parser.add_argument("--automatic-only", action="store_true")
    parser.add_argument("--reject-field", action="append", choices=FIELDS, default=[])
    parser.add_argument("--output", type=Path, default=ROOT / "build/handwriting/scores")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    labels_path = DATA / "labels.csv"
    labels_hash = hashlib.sha256(labels_path.read_bytes()).hexdigest()
    labels = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
    labels = labels.loc[labels.split.eq(args.split)].copy()
    page_runs = pd.read_csv(args.predictions / f"{args.split}_pages.csv",
                            keep_default_na=False).set_index("item_id")
    audits = json.loads((DATA / "layout_audit.json").read_text())["paris"]
    cell_records, page_records = [], []
    for page_id, truth in labels.groupby("page_id", sort=False):
        response_bytes = (args.predictions / f"{page_id}.html").read_bytes()
        html = response_bytes.decode()
        n_tables = len(BeautifulSoup(html, "html.parser").find_all("table"))
        rows = html_rows(html)
        complete = page_runs.loc[page_id, "finish_reason"] == "stop"
        row_count_matches = n_tables == 1 and len(rows) == len(truth) and complete
        audit = audits.get(hashlib.sha256(response_bytes).hexdigest())
        if audit and (audit["reference_sha256"] != labels_hash
                      or audit["image_sha256"] != hashlib.sha256(
                          (DATA / "pages" / f"{page_id}.png").read_bytes()).hexdigest()):
            print(f"{page_id}: recorded audit does not match source hashes; automatic alignment only")
            audit = None
        positions = list(range(len(rows))) if row_count_matches else []
        assisted = not positions and audit is not None and not args.automatic_only
        if assisted:
            positions = audit["reference_row_positions"]
        projected = {position: rows[i] for i, position in enumerate(positions)}
        page_cells = []
        for i, row in enumerate(truth.itertuples()):
            for j, field in enumerate(FIELDS):
                reference = normalize(getattr(row, field).replace("!", " ").replace("?", " "))
                prediction = html_value(projected.get(i, {}), [j])
                if field in args.reject_field:
                    prediction = None
                if prediction is not None:
                    prediction = normalize(prediction)
                score = cell_score(reference, prediction, field == "birth_year")
                page_cells.append({
                    "row_id": row.row_id, "page_id": page_id, "field": field,
                    "reference": reference, "prediction": prediction,
                    "automatic_available": prediction is not None and not assisted,
                    "automatic_correct": score["correct"] if not assisted else None,
                    "automatic_character_errors":
                    score["character_errors"] if not assisted else 0,
                    "assisted": prediction is not None and assisted,
                    **score,
                })
        cell_records.extend(page_cells)
        page_records.append({
            "page_id": page_id, "expected_rows": len(truth),
            "predicted_rows": len(rows),
            "predicted_columns": max((len(row) for row in rows), default=0),
            "tables": n_tables, "complete": complete,
            "row_count_matches": row_count_matches,
            "automatic_rows": len(positions) if not assisted else 0,
            "available_rows": len(positions),
            "assisted_rows": len(positions) if assisted else 0,
            "available_cells": sum(c["available"] for c in page_cells),
            "automatic_cells": sum(c["automatic_available"] for c in page_cells),
            "alignment": "source-audited" if assisted else "automatic",
        })
    cells = pd.DataFrame(cell_records)
    cells.to_csv(args.output / "cells.csv", index=False, na_rep="NaN")
    cells.pivot(index=["page_id", "row_id"], columns="field", values="prediction").reindex(
        columns=FIELDS).to_csv(args.output / "predictions.csv", na_rep="NaN")
    cells.assign(prediction=cells.prediction.where(~cells.assisted)).pivot(
        index=["page_id", "row_id"], columns="field", values="prediction").reindex(
        columns=FIELDS).to_csv(args.output / "automatic_predictions.csv", na_rep="NaN")
    pd.DataFrame(page_records).to_csv(args.output / "structure.csv", index=False)
    field_scores = [{"field": field, **cell_metrics(group)}
                    for field, group in cells.groupby("field", sort=False)]
    pd.DataFrame(field_scores).to_csv(args.output / "field_scores.csv", index=False,
                                     na_rep="NaN")
    summary = {
        "split": args.split, "rows": len(labels), "pages": labels.page_id.nunique(),
        **cell_metrics(cells),
        "exact_table_rows": int(cells.assign(exact=cells.correct.eq(1))
                                .groupby("row_id").exact.all().sum()),
        "count_matched_pages": sum(row["row_count_matches"] for row in page_records),
        "automatic_only": args.automatic_only,
        "alignment": "source-audited" if cells.assisted.any() else "automatic",
        "rejected_fields": ",".join(args.reject_field),
    }
    if args.pages_only:
        pd.DataFrame([summary]).to_csv(args.output / "summary.csv", index=False,
                                       na_rep="NaN")
        print(pd.DataFrame(page_records).to_string(index=False))
        print(pd.DataFrame(field_scores).to_string(index=False))
        return
    row_runs = pd.read_csv(args.predictions / f"{args.split}_rows.csv",
                           dtype=str, keep_default_na=False).set_index("item_id")
    line_records = []
    for row in labels.itertuples():
        reference = normalize(row.source_text.translate(str.maketrans({
            "/": " ", "¤": "", "!": " ", "?": " ",
        })))
        prediction = normalize(row_runs.loc[row.row_id, "text"])
        line_records.append({
            "row_id": row.row_id, "page_id": row.page_id,
            "reference": reference, "prediction": prediction,
            "finish_reason": row_runs.loc[row.row_id, "finish_reason"],
            "characters": len(reference), "words": len(reference.split()),
            "character_errors": edit_distance(reference, prediction),
            "word_errors": edit_distance(reference.split(), prediction.split()),
            "exact": reference == prediction,
        })
    lines = pd.DataFrame(line_records)
    lines.to_csv(args.output / "lines.csv", index=False)
    summary.update({
        "cer": lines.character_errors.sum() / lines.characters.sum(),
        "wer": lines.word_errors.sum() / lines.words.sum(),
        "exact_lines": int(lines.exact.sum()),
        "truncated_lines": int(lines.finish_reason.ne("stop").sum()),
    })
    pd.DataFrame([summary]).to_csv(args.output / "summary.csv", index=False,
                                   na_rep="NaN")
    print(pd.DataFrame([summary]).to_string(index=False))
    print(pd.DataFrame(field_scores).to_string(index=False))


if __name__ == "__main__":
    main()
