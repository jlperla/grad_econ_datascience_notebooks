import argparse
import json
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "slides/data/handwriting"
FIELDS = [
    "surname", "given_names", "birth_year", "birthplace", "nationality",
    "marital_status", "relationship", "education", "occupation", "employer",
]
PROMPT = (DATA / "claude_prompt.txt").read_text()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "build/handwriting/claude")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pages = pd.read_csv(DATA / "pages.csv")
    records, models = [], set()
    for page in pages.loc[pages.split.eq("test")].itertuples():
        with tempfile.TemporaryDirectory() as workdir:
            shutil.copyfile(DATA / page.image, Path(workdir) / "census.png")
            result = subprocess.run(
                ["claude", "-p", "--model", "opus", "--output-format", "json",
                 "--tools", "Read", "--allowedTools", "Read"],
                input=PROMPT + "census.png", text=True,
                capture_output=True, check=True, cwd=workdir,
            )
        audit_dir = ROOT / "build/handwriting/claude_raw"
        audit_dir.mkdir(parents=True, exist_ok=True)
        (audit_dir / f"{page.page_id}.json").write_text(result.stdout)
        response = json.loads(result.stdout)
        models.update(response["modelUsage"])
        raw = response["result"]
        rows = json.loads(raw.removeprefix("```json").removesuffix("```").strip())
        table = pd.DataFrame(rows, columns=FIELDS)
        (args.output / f"{page.page_id}.html").write_text(table.to_html(index=False))
        (args.output / f"{page.page_id}.json").write_text(json.dumps({
            "result": raw, "models": list(response["modelUsage"]),
            "image_sha256": page.image_sha256,
            "num_turns": response["num_turns"],
            "permission_denials": response["permission_denials"],
        }, ensure_ascii=False, indent=2) + "\n")
        records.append({
            "item_id": page.page_id, "page_id": page.page_id, "split": "test",
            "finish_reason": "stop" if not response["is_error"] else "error",
        })
        print(page.page_id, len(table), flush=True)
    pd.DataFrame(records).to_csv(args.output / "test_pages.csv", index=False)
    (args.output / "test_pages_run.json").write_text(json.dumps({
        "models": sorted(models), "date_utc": datetime.now(UTC).isoformat(),
        "prompt": PROMPT, "input": "Same lossless page crops as Chandra",
        "input_isolation": "Only census.png copied into a temporary directory",
        "tools": ["Read"], "labels_supplied": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
