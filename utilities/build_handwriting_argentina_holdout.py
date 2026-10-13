import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "build/handwriting/argentina/holdout"
PAGE_ID = "S3HT-62RW-C1Y"
THESIS_URL = (
    "https://odr.chalmers.se/bitstreams/"
    "56e9e0b8-2859-4a1c-8d3c-735b0441de16/download"
)


def download_image():
    CACHE.mkdir(parents=True, exist_ok=True)
    thesis = CACHE / "thesis.pdf"
    if not thesis.exists():
        thesis.write_bytes(urlopen(THESIS_URL).read())
    subprocess.run([
        "pdfimages", "-f", "64", "-l", "64", "-png", str(thesis),
        str(CACHE / "source"),
    ], check=True)
    image = (CACHE / "source-000.png").read_bytes()
    metadata = {
        "source": "Argentina 1895 census, held-out page S3HT-62RW-C1Y",
        "source_url": THESIS_URL, "filename": "Figure 5.16, PDF page 64",
        "thesis_sha256": hashlib.sha256(thesis.read_bytes()).hexdigest(),
        "source_extraction": "Poppler pdfimages; embedded image, no screenshot",
        "source_preprocessing": "Author's median blur, neighborhood 5",
        "split": "author final test set",
        "split_source": THESIS_URL + "#page=62",
        "selection": "Author selected the best of 14 final test pages",
        "evaluation": "One selected held-out page; not representative accuracy",
    }
    (CACHE / "source.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return image, metadata


def reference_rows():
    subprocess.run([
        "pdftotext", "-f", "92", "-l", "92", "-layout",
        str(CACHE / "thesis.pdf"), str(CACHE / "appendix.txt"),
    ], check=True)
    text = (CACHE / "appendix.txt").read_text()
    block = text.split("Figure A.30:")[1].split("Figure A.31:")[0]
    # TeX renders adjacent comma delimiters as a low double-quote glyph.
    rows = [line.strip().replace("„", ",,").split(",")
            for line in block.splitlines()
            if re.match(r"^(P1-)?,[0-9]+,", line.strip())]
    with (CACHE / "reference.csv").open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)
    return rows


if __name__ == "__main__":
    download_image()
    rows = reference_rows()
    print(f"{len(rows)} reference people; source and labels in {CACHE}")
