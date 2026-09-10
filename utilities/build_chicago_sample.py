# Random sample of full-time salaried positions from the City of Chicago
# "Current Employee Names, Salaries, and Position Titles" dataset on the
# Chicago Data Portal (data.cityofchicago.org, dataset xzkq-xp2w). Names dropped.
# Rebuild: uv run python utilities/build_chicago_sample.py
import io
import requests
import pandas as pd

URL = "https://data.cityofchicago.org/resource/xzkq-xp2w.csv?$limit=50000"

r = requests.get(URL, timeout=120)
r.raise_for_status()
raw = pd.read_csv(io.StringIO(r.text))
full = raw[(raw["full_or_part_time"] == "F") & (raw["salary_or_hourly"] == "SALARY")]
df = full.sample(8000, random_state=526).rename(columns={"job_titles": "job_title"})
df = df[["job_title", "department", "annual_salary"]].reset_index(drop=True)
df.to_csv("slides/data/chicago_payroll.csv.gz", index=False)
print(raw.shape, full.shape, df.shape, df["job_title"].nunique(), "titles")
