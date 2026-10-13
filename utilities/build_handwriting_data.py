import hashlib
import json
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import fsspec
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "slides/data/handwriting"
REVISION = "580000d295c88e02bb7d8578ed203b7cde5deba7"
SOURCE = f"https://raw.githubusercontent.com/constum-thomas-m/POPP-datasets/{REVISION}"
ARCHIVE = "https://zenodo.org/api/records/6581158/files/popp.zip/content"
PAGES = {
    "development": ["275_0025", "301_0020"],
    "test": ["245_0008", "235_0105", "266_0030", "223_0430"],
}
FIELDS = [
    "surname", "given_names", "birth_year", "birthplace", "nationality",
    "marital_status", "relationship", "education", "occupation", "employer",
]


def main():
    (DEST / "pages").mkdir(parents=True, exist_ok=True)
    with urlopen(f"{SOURCE}/Generic/lines/labels.json") as response:
        labels = json.load(response)["ground_truth"]
    rows, pages = [], []
    with (
        fsspec.open(ARCHIVE, block_size=1024 * 1024) as remote,
        ZipFile(remote) as archive,
    ):
        for split, ids in PAGES.items():
            source_split = "valid" if split == "development" else "test"
            for short_id in ids:
                page_id = f"AD075DP_D2M8_{short_id}_left_page"
                folder = "data/Generic/pages"
                xml = ET.fromstring(archive.read(f"{folder}/xml/{page_id}.alto.xml"))
                lines = xml.findall(".//{*}TextLine")
                image = Image.open(BytesIO(archive.read(
                    f"{folder}/images/{page_id}.tif"
                ))).convert("RGB")
                left = min(int(line.attrib["HPOS"]) for line in lines)
                image = image.crop((left, 0, image.width, image.height))
                image_path = DEST / "pages" / f"{page_id}.png"
                image.save(image_path)
                pages.append({
                    "page_id": page_id, "split": split,
                    "image": f"pages/{page_id}.png", "crop_left": left,
                    "width": image.width, "height": image.height,
                    "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                })
                for row_number, line in enumerate(lines):
                    row_id = f"{page_id}_{row_number:02d}"
                    text = labels[source_split][f"{row_id}.tif"]["text"]
                    fields = [" ".join(cell.replace("¤", "").split())
                              for cell in text.split("/")]
                    x, y = int(line.attrib["HPOS"]) - left, int(line.attrib["VPOS"])
                    width, height = int(line.attrib["WIDTH"]), int(line.attrib["HEIGHT"])
                    rows.append({
                        "row_id": row_id, "page_id": page_id, "split": split,
                        "row_number": row_number, "x": x, "y": y,
                        "width": width, "height": height, "source_text": text,
                        **dict(zip(FIELDS, fields, strict=True)),
                    })
                print(split, page_id, len(lines), flush=True)
    pd.DataFrame(pages).to_csv(DEST / "pages.csv", index=False)
    pd.DataFrame(rows).to_csv(DEST / "labels.csv", index=False)
    (DEST / "source.json").write_text(json.dumps({
        "dataset": "POPP Generic, 1926 Paris census",
        "doi": "10.5281/zenodo.6581158", "license": "CC-BY-4.0",
        "labels_revision": REVISION, "archive": ARCHIVE,
        "build_date": datetime.now(UTC).date().isoformat(), "test_selection_seed": 526,
        "selection": "Four of eight test double-page IDs sampled with random.Random(526).sample(sorted(ids), 4); left pages only.",
        "development_selection": "Two manually selected validation pages: 275_0025 and 301_0020.",
        "images": "Lossless crop to columns 6–15; original printed headers retained.",
    }, indent=2) + "\n")
    (DEST / "ATTRIBUTION.txt").write_text(
        "POPP Generic dataset: 1926 Paris census.\n"
        "Constum, Kempf, Paquet, Tranouez, Chatelain, Brée, and Merveille (2022).\n"
        "Recognition and Information Extraction in Historical Handwritten Tables:\n"
        "Toward Understanding Early 20th Century Paris Census. DAS, pp. 143–157.\n"
        "Source: https://doi.org/10.5281/zenodo.6581158\n"
        "License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/\n"
        "Changes: six pages cropped to columns 6–15 and stored as lossless PNG;\n"
        "180 source transcriptions parsed into ten fields with whitespace collapsed.\n"
        "Original annotations and source coordinates are retained in labels.csv.\n"
        "Model responses are predictions, not source transcriptions.\n"
    )


if __name__ == "__main__":
    main()
