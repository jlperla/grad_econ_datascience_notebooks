# Census commuting flows, county boundaries, and reference coordinates.
# Rebuild: uv run --extra extras python utilities/build_spatial_data.py
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "slides" / "data"
DATA.mkdir(exist_ok=True)
CENSUS = "https://www2.census.gov"
FLOWS_URL = (
    f"{CENSUS}/programs-surveys/demo/tables/metro-micro/2020/"
    "commuting-flows-2020/table1.xlsx"
)
COUNTIES_URL = f"{CENSUS}/geo/tiger/GENZ2020/shp/cb_2020_us_county_500k.zip"
PLACES_URL = (
    f"{CENSUS}/geo/docs/maps-data/data/gazetteer/2020_Gazetteer/2020_gaz_place_53.txt"
)
CENTERS_URL = f"{CENSUS}/geo/docs/reference/cenpop2020/county/CenPop2020_Mean_CO.txt"
FLOW_COLUMNS = [
    "home_state",
    "home_county",
    "home_state_name",
    "home_name",
    "work_state",
    "work_county",
    "work_state_name",
    "work_name",
    "workers",
    "moe",
]
CODE_COLUMNS = ["home_state", "home_county", "work_state", "work_county"]

with TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    urlretrieve(FLOWS_URL, tmp / "flows.xlsx")
    flows = pd.read_excel(
        tmp / "flows.xlsx",
        skiprows=8,
        skipfooter=4,
        header=None,
        names=FLOW_COLUMNS,
        dtype=dict.fromkeys(CODE_COLUMNS, str),
    )
    flows.to_csv(
        DATA / "spatial_commuting.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )

    urlretrieve(PLACES_URL, tmp / "places.txt")
    places = pd.read_csv(
        tmp / "places.txt",
        sep="\t",
        dtype={"GEOID": str, "ANSICODE": str, "LSAD": str},
    )
    places.columns = places.columns.str.strip()
    places = places.rename(
        columns={"NAME": "city", "INTPTLONG": "lon", "INTPTLAT": "lat"}
    )
    places.to_csv(DATA / "spatial_places.csv", index=False)

    urlretrieve(CENTERS_URL, tmp / "centers.txt")
    population = pd.read_csv(
        tmp / "centers.txt", dtype={"STATEFP": str, "COUNTYFP": str}
    ).rename(columns={"LONGITUDE": "pop_lon", "LATITUDE": "pop_lat"})
    population.to_csv(DATA / "spatial_population_centers.csv", index=False)

urlretrieve(COUNTIES_URL, DATA / "spatial_counties.zip")

print(f"Saved four Census input files in {DATA}")
