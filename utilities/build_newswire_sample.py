# Stratified random sample of the Newswire dataset drawn by HTTP range reads of
# the yearly JSON files on the Hugging Face Hub (35 GB in total, so the files
# are never downloaded whole). CC BY 4.0, DOI 10.57967/hf/2423.
# Silcock, Arora, D'Amico-Wong, Dell (2024), "Newswire: A Large-Scale Structured
# Database of a Century of Historical News", arXiv:2406.09490.
# Rebuild: uv run python utilities/build_newswire_sample.py
import ast
import csv
import gzip
import json
import random
import sys
import requests

BASE = ("https://huggingface.co/datasets/dell-research-harvard/newswire"
        "/resolve/main/{y}_data_clean.json")
SEP = "\n    },\n    {\n"
LABELS = ["politics", "civil_rights", "labor_movement", "antitrust",
          "govt_regulation", "protests", "crime"]
WINDOW = 1_000_000


def file_size(y):
    r = requests.head(BASE.format(y=y), allow_redirects=True, timeout=60)
    r.raise_for_status()
    return int(r.headers["content-length"])


def window_records(y, offset):
    r = requests.get(BASE.format(y=y),
                     headers={"Range": f"bytes={offset}-{offset + WINDOW - 1}"},
                     timeout=180)
    r.raise_for_status()
    fragments = r.text.split(SEP)[1:-1]
    return [json.loads("{\n" + frag + "\n    }") for frag in fragments]


def state_of(field):
    if field.lstrip().startswith("("):
        v = ast.literal_eval(field)
        while isinstance(v, tuple):
            v = v[0]
        return v.strip().lower()
    return field.strip().lower()


def sample_year(y, target, rng):
    n, kept, seen = file_size(y), [], set()
    while len(kept) < target:
        for a in window_records(y, rng.randrange(0, n - WINDOW)):
            t = a["cleaned_article"]
            if t and t not in seen:
                seen.add(t)
                kept.append(a)
    return rng.sample(kept, target)


def row(a):
    states = [state_of(m["newspaper_state"]) for m in a["newspaper_metadata"]]
    return ([a["year"], a["dates"][0], a["cleaned_article"], a["ca_topic"],
             a["wire_state"].strip(), a["wire_country"].strip(),
             len(states), ";".join(states)]
            + [a[k] for k in LABELS])


rng = random.Random(526)
plan = [(y, 40) for y in range(1900, 1947)] + [(y, 400) for y in range(1947, 1978)]
with gzip.open("slides/data/newswire_sample.csv.gz", "wt",
               newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["year", "date", "text", "ca_topic", "wire_state", "wire_country",
                "n_papers", "paper_states"] + LABELS)
    for y, target in plan:
        for a in sample_year(y, target, rng):
            w.writerow(row(a))
        print(y, file=sys.stderr, flush=True)
