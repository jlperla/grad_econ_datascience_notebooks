import argparse
import json
import re
import unicodedata
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path

import pandas as pd
from build_handwriting_scores import edit_distance

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["age", "earnings"]


def normalize(value, field):
    if pd.isna(value):
        return float("nan")
    value = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    if value in {"", "*", "-", "—", "–"}:
        return ""
    value = value.replace(",", "").replace("$", "").strip()
    try:
        if field == "age" and re.fullmatch(r"\d+\s*/\s*\d+", value):
            return str(int(Fraction(value.replace(" ", ""))))
        number = Decimal(value)
        if field == "earnings":
            number = number.quantize(Decimal(1), rounding=ROUND_HALF_UP)
        return format(number.normalize(), "f")
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return " ".join(value.split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "build/handwriting/toronto")
    parser.add_argument("--predictions", type=Path,
                        default=ROOT / "build/handwriting/toronto")
    parser.add_argument("--split", choices=["development", "test"], default="test")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.data / "scores"
    output.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(args.data / "labels.csv", dtype=str, keep_default_na=False)
    labels = labels.loc[labels.split.eq(args.split)]
    cells, structures, earnings_summaries = [], [], []
    for workflow in ["chandra", "qwen27b", "qwen2b"]:
        predictions = pd.read_csv(args.predictions / workflow /
                                  f"{args.split}_predictions.csv", dtype=str,
                                  keep_default_na=False, na_values=["NaN"])
        structure = pd.read_csv(args.predictions / workflow /
                                f"{args.split}_structure.csv")
        structures.append({"workflow": workflow, "panels": len(structure),
                           "age_available": int(structure.age_available.sum()),
                           "earnings_available": int(structure.earnings_available.sum()),
                           "expected_cells_per_field": len(structure) * 10,
                           "rows_emitted": int(structure.rows_emitted.sum())})
        joined = labels.merge(predictions, on=["page_id", "line"], how="left",
                              suffixes=("_ref", "_pred"))
        annual = joined.loc[joined.earnings_period.eq("")].copy()
        annual["reference_dollars"] = pd.to_numeric(
            annual.earnings_ref.map(lambda x: normalize(x, "earnings")),
            errors="coerce")
        annual["predicted_dollars"] = pd.to_numeric(
            annual.earnings_pred.map(lambda x: normalize(x, "earnings")),
            errors="coerce")
        annual = annual.loc[annual.reference_dollars.gt(0)]
        available = annual.loc[annual.predicted_dollars.notna()]
        differences = available.predicted_dollars - available.reference_dollars
        earnings_summaries.append({
            "workflow": workflow, "reference_annual_earners": len(annual),
            "numeric_predictions": len(available),
            "reference_mean": annual.reference_dollars.mean(),
            "paired_reference_mean": available.reference_dollars.mean(),
            "predicted_mean": available.predicted_dollars.mean(),
            "paired_mean_error": differences.mean(),
            "paired_mean_absolute_error": differences.abs().mean(),
        })
        for person in joined.to_dict("records"):
            for field in FIELDS:
                reference = person[f"{field}_ref"]
                if field == "age" and reference == "" and person["age_months"]:
                    reference = person["age_months"] + "/12"
                prediction = person[f"{field}_pred"]
                uncertain = bool(re.search(r"[!?{}=]", reference))
                ref_norm, pred_norm = normalize(reference, field), normalize(prediction, field)
                available = pd.notna(prediction)
                reference_digits = len(ref_norm) if ref_norm.isdecimal() else 0
                cells.append({
                    "workflow": workflow, "page_id": person["page_id"],
                    "line": person["line"], "field": field,
                    "reference": reference, "prediction": prediction,
                    "reference_normalized": ref_norm, "prediction_normalized": pred_norm,
                    "reference_uncertain": uncertain, "filled": ref_norm != "",
                    "available": available,
                    "correct": ref_norm == pred_norm if available else float("nan"),
                    "reference_digits": reference_digits,
                    "digit_errors": edit_distance(ref_norm, pred_norm)
                    if available and reference_digits else float("nan"),
                })
    frame = pd.DataFrame(cells)
    frame.to_csv(output / f"{args.split}_cells.csv", index=False, na_rep="NaN")
    audit = json.loads((ROOT / "slides/data/handwriting/toronto/audit.json").read_text())
    difficult_page = audit["post_run_age_diagnostic_page_id"]
    flagged = {(flag["page_id"], str(line))
               for flag in audit["pre_inference_label_flags"]
               if flag["field"] == "age" for line in flag["lines"]}
    ages = frame.loc[frame.workflow.eq("chandra") & frame.field.eq("age")].copy()
    ages["reason"] = [
        "Unavailable age" if not row.available else
        "Hutchman page" if row.page_id == difficult_page else
        "Flagged reference" if (row.page_id, row.line) in flagged else
        "Other digit disagreements" for row in ages.itertuples()
    ]
    missing_earnings = frame.loc[frame.workflow.eq("chandra")
                                & frame.field.eq("earnings") & ~frame.available,
                                ["page_id", "line"]]
    retained = ages.loc[ages.available].merge(missing_earnings, on=["page_id", "line"])
    page_ages = ages.loc[ages.page_id.eq(difficult_page)]
    (output / f"{args.split}_age_audit.json").write_text(json.dumps({
        "workflow": "chandra", "people": len(ages),
        "available": int(ages.available.sum()),
        "correct": int(ages.correct.sum()),
        "disagreements": ages.loc[ages.correct.eq(False)].groupby("reason").size().to_dict(),
        "ages_retained_with_unavailable_earnings": len(retained),
        "correct_ages_with_unavailable_earnings": int(retained.correct.sum()),
        "difficult_page": difficult_page, "page_people": len(page_ages),
        "page_correct": int(page_ages.correct.sum()),
        "interpretation": "Whole-age disagreements among available cells. A failed earnings column does not discard ages. Flagged references are unresolved, not corrected labels.",
    }, indent=2) + "\n")
    summaries = []
    for keys, group in frame.groupby(["workflow", "field"], sort=False):
        known = group.loc[~group.reference_uncertain]
        filled, blank = known.loc[known.filled], known.loc[~known.filled]
        readable = filled.loc[filled.available]
        digits = readable.loc[readable.reference_digits.gt(0)]
        digit_count = int(digits.reference_digits.sum())
        digit_errors = int(digits.digit_errors.sum())
        summaries.append({
            "workflow": keys[0], "field": keys[1], "reference_cells": len(group),
            "uncertain_references": int(group.reference_uncertain.sum()),
            "filled_cells": len(filled), "filled_correct": int(filled.correct.sum()),
            "filled_available": len(readable),
            "filled_agreement": filled.correct.mean(),
            "filled_coverage": filled.available.mean(),
            "reference_digits": digit_count, "digit_errors": digit_errors,
            "digit_accuracy": 1 - digit_errors / digit_count if digit_count else float("nan"),
            "blank_cells": len(blank), "blank_available": int(blank.available.sum()),
            "blank_correct": int(blank.correct.sum()),
            "blank_agreement": blank.correct.mean(),
            "unavailable_cells": int((~group.available).sum()),
        })
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / f"{args.split}_field_scores.csv", index=False, na_rep="NaN")
    pd.DataFrame(structures).to_csv(output / f"{args.split}_structure.csv", index=False)
    pd.DataFrame(earnings_summaries).to_csv(
        output / f"{args.split}_earnings.csv", index=False)
    page_scores = frame.loc[~frame.reference_uncertain & frame.filled].groupby(
        ["workflow", "page_id", "field"]).agg(cells=("correct", "size"),
                                               available=("available", "sum"),
                                               correct=("correct", "sum"))
    page_scores.to_csv(output / f"{args.split}_page_scores.csv")
    (output / f"{args.split}_sample.json").write_text(json.dumps({
        "people": len(labels), "pages": labels.page_id.nunique(),
        "enumerators": labels.enumerator.nunique(), "split": args.split,
        "metrics": "Digit accuracy = 1 - summed character edit distance / reference digits, using available filled cells after numeric normalization. Whole-value agreement and extraction coverage are separate.",
        "missing_output": "NaN marks an unavailable field; an extracted empty string remains a blank prediction. NaN cells are excluded from conditional accuracy and counted in coverage.",
        "scored_fields": FIELDS,
        "unscored_fields": ["name", "occupation"],
        "unscored_reason": "Names can contain expanded surnames; occupations mix abbreviations and editorial expansions without an adjudicated equivalence mapping.",
        "uncertainty": "References containing ! ? { } = are reported separately",
    }, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
