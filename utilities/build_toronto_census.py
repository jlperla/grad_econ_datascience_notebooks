import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
import pyreadstat
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "slides/data/handwriting/toronto"
CACHE = ROOT / "build/handwriting/toronto"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=CACHE)
    args = parser.parse_args()
    source = json.loads((DATA / "source.json").read_text())
    pages = json.loads((DATA / "pages.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    sav = args.output / "toronto-1901.sav"
    if not sav.exists():
        subprocess.run(["curl", "-L", "--fail", "--show-error", "--silent",
                        source["labels_url"], "-o", str(sav)], check=True)
    people, _ = pyreadstat.read_sav(sav)
    for column in people.select_dtypes(include=["str", "object"]):
        people[column] = people[column].str.strip()
    labels, panels, downloads = [], [], []
    font = ImageFont.load_default(size=16)
    widths = {"name": 270, "age": 80, "occupation": 200, "earnings": 90}
    for page in pages:
        page_id = page["page_id"]
        image_path = args.output / f"{page_id}.jpg"
        if not image_path.exists():
            subprocess.run(["curl", "-L", "--fail", "--show-error", "--silent",
                            page["image_url"], "-o", str(image_path)], check=True)
        image = Image.open(image_path).convert("RGB")
        rectified = image.transform(tuple(page["rectified_size"]), Image.Transform.QUAD,
                                    page["quad"], Image.Resampling.BICUBIC)
        selected = people
        for column, value in page["york_key"].items():
            selected = selected.loc[selected[column].eq(value)]
        selected = selected.sort_values("LineNo", key=lambda s: pd.to_numeric(s))
        for person in selected.to_dict("records"):
            labels.append({
                "page_id": page_id, "split": page["split"],
                "line": int(person["LineNo"]),
                "surname": person["LastName"], "given_names": person["FirstName"],
                "age": person["Ageyr"], "age_months": person["Agemo"],
                "occupation": person["OCC"], "earnings": person["EARNINGS"],
                "earnings_period": person["EARNSPER"],
                "individual_notes": person["Individual_NOTES"],
                "dwelling_note": person["DwellingNote"],
                "enumerator": page["enumerator"],
            })
        for first_line in range(1, 51, 10):
            panel_id = f"{page_id}_{first_line:02d}-{first_line + 9:02d}"
            panel = Image.new("RGB", (705, 535), "white")
            draw = ImageDraw.Draw(panel)
            draw.text((8, 9), "Line", fill="black", font=font)
            left = 55
            for field, width in widths.items():
                draw.text((left + 5, 9), field.title(), fill="black", font=font)
                column = page["columns"][field]
                for offset in range(10):
                    line = first_line + offset
                    top, bottom = column["row_edges"][line - 1:line + 1]
                    cell = rectified.crop((column["left"], top,
                                           column["right"], bottom))
                    panel.paste(cell.resize((width, 50), Image.Resampling.LANCZOS),
                                (left, 35 + offset * 50))
                draw.line((left, 0, left, 535), fill="black")
                left += width
            for offset in range(10):
                top = 35 + offset * 50
                draw.text((14, top + 15), str(first_line + offset),
                          fill="black", font=font)
                draw.line((0, top, 705, top), fill="black")
            panel = panel.resize((1410, 1070), Image.Resampling.LANCZOS)
            panel_path = args.output / f"{panel_id}.png"
            panel.save(panel_path)
            panels.append({
                "panel_id": panel_id, "page_id": page_id, "split": page["split"],
                "first_line": first_line, "last_line": first_line + 9,
                "image": panel_path.name,
                "sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest(),
            })
        downloads.append({"page_id": page_id, "image_url": page["image_url"],
                          "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest()})
    pd.DataFrame(labels).to_csv(args.output / "labels.csv", index=False)
    pd.DataFrame(panels).to_csv(args.output / "panels.csv", index=False)
    (args.output / "downloads.json").write_text(json.dumps({
        "labels_url": source["labels_url"], "labels_rows": len(people),
        "labels_sha256": hashlib.sha256(sav.read_bytes()).hexdigest(),
        "pages": downloads,
    }, indent=2) + "\n")
    print(pd.DataFrame(labels).groupby(["split", "page_id"]).size().to_string())
    print(f"Images, panels and reference labels: {args.output}")


if __name__ == "__main__":
    main()
