# BLS major work stoppages (1,000 workers or more), annual listing, 1947-1977.
# Source: https://www.bls.gov/web/wkstp/annual-listing.htm (public domain).
# bls.gov refuses requests from outside the United States, so this reads the
# Internet Archive capture of the same page.
# Rebuild: uv run --with lxml python utilities/build_work_stoppages.py
import gzip
import io
import requests
import pandas as pd

URL = ("https://web.archive.org/web/20251231013105id_/"
       "https://www.bls.gov/web/wkstp/annual-listing.htm")

r = requests.get(URL, timeout=120)
r.raise_for_status()
body = r.content
if body[:2] == b"\x1f\x8b":
    body = gzip.decompress(body)
table = pd.read_html(io.StringIO(body.decode("utf-8")))[0]
table.columns = ["year", "stoppages", "stoppages_in_effect", "workers_thousands",
                 "workers_in_effect", "days_idle_thousands", "pct_working_time"]
df = table[table["year"].astype(str).str.fullmatch(r"\d{4}")]
df = df[["year", "stoppages", "workers_thousands", "days_idle_thousands"]]
df = df.astype({"year": int, "stoppages": int, "workers_thousands": float,
                "days_idle_thousands": float})
df = df[(df["year"] >= 1947) & (df["year"] <= 1977)]
df.to_csv("slides/data/work_stoppages.csv", index=False)
print(df.shape)
print(df.head(3).to_string(index=False))
print(df.tail(3).to_string(index=False))
