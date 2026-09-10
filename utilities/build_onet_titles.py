# O*NET 31.0 sample of reported job titles with their SOC code and major group.
# O*NET is developed under the sponsorship of the U.S. Department of Labor,
# Employment and Training Administration (USDOL/ETA), CC BY 4.0.
# Rebuild: uv run python utilities/build_onet_titles.py
import io
import requests
import pandas as pd

URL = ("https://www.onetcenter.org/dl_files/database/db_31_0_csv/"
       "sample_of_reported_titles.csv")
MAJOR_GROUPS = {
    "11": "Management",
    "13": "Business and Financial Operations",
    "15": "Computer and Mathematical",
    "17": "Architecture and Engineering",
    "19": "Life, Physical, and Social Science",
    "21": "Community and Social Service",
    "23": "Legal",
    "25": "Educational Instruction and Library",
    "27": "Arts, Design, Entertainment, Sports, and Media",
    "29": "Healthcare Practitioners and Technical",
    "31": "Healthcare Support",
    "33": "Protective Service",
    "35": "Food Preparation and Serving Related",
    "37": "Building and Grounds Cleaning and Maintenance",
    "39": "Personal Care and Service",
    "41": "Sales and Related",
    "43": "Office and Administrative Support",
    "45": "Farming, Fishing, and Forestry",
    "47": "Construction and Extraction",
    "49": "Installation, Maintenance, and Repair",
    "51": "Production",
    "53": "Transportation and Material Moving",
    "55": "Military Specific",
}

r = requests.get(URL, timeout=60)
r.raise_for_status()
raw = pd.read_csv(io.StringIO(r.text))
df = pd.DataFrame({
    "title": raw["Reported Job Title"],
    "soc_code": raw["O*NET-SOC Code"],
    "soc_title": raw["Title"],
    "major_group": raw["O*NET-SOC Code"].str[:2],
})
df["major_group_title"] = [MAJOR_GROUPS[g] for g in df["major_group"]]
df.to_csv("slides/data/onet_titles.csv.gz", index=False)
print(df.shape, df["major_group"].nunique(), "major groups")
