"""Download 2025 earthquakes (M2.5+) from the USGS Earthquake Catalog.

Source: https://earthquake.usgs.gov/fdsnws/event/1/ (USGS data is public domain).
The API returns at most 20,000 events per query, so this asks for one month at a time.
"""
import io

import pandas as pd
import requests

URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
COLUMNS = ["time", "latitude", "longitude", "depth", "mag", "magType", "nst", "gap",
           "dmin", "rms", "place", "type", "status"]
OUT = "data/usgs_earthquakes_2025.csv"


def main():
    frames = []
    for month in range(1, 13):
        start = f"2025-{month:02d}-01"
        end = f"2025-{month + 1:02d}-01" if month < 12 else "2026-01-01"
        response = requests.get(URL, params={
            "format": "csv", "starttime": start, "endtime": end,
            "minmagnitude": 2.5, "orderby": "time-asc", "limit": 20000,
        }, timeout=120)
        response.raise_for_status()
        part = pd.read_csv(io.StringIO(response.text))
        print(f"{start}: {len(part)} events")
        assert len(part) < 20000, "hit the API row cap, split the query further"
        frames.append(part[COLUMNS])

    df = pd.concat(frames, ignore_index=True)
    df = df[df["type"] == "earthquake"].drop(columns="type")  # drop quarry blasts etc.
    df = df.drop_duplicates()
    df.to_csv(OUT, index=False)
    print(f"saved {OUT}: {len(df)} earthquakes, {df['time'].min()} -> {df['time'].max()}")
    print(f"missing values per column:\n{df.isna().sum().to_string()}")


if __name__ == "__main__":
    main()
