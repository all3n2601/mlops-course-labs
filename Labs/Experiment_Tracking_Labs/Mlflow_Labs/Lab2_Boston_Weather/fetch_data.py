"""Download hourly Boston weather (2023-2025) from the Open-Meteo archive API.

Data source: https://open-meteo.com (historical weather API, reanalysis data, CC BY 4.0).
Coordinates are the Northeastern University campus. Times are stored in UTC so every
row is exactly one hour after the previous one.
"""
import pandas as pd
import requests

URL = "https://archive-api.open-meteo.com/v1/archive"
VARIABLES = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "pressure_msl",
    "cloud_cover", "wind_speed_10m", "wind_direction_10m", "precipitation",
    "shortwave_radiation",
]
OUT = "data/boston_hourly_2023_2025.csv"


def main():
    params = {
        "latitude": 42.3398,
        "longitude": -71.0892,
        "start_date": "2023-01-01",
        "end_date": "2025-12-31",
        "hourly": ",".join(VARIABLES),
        "timezone": "GMT",
    }
    response = requests.get(URL, params=params, timeout=60)
    response.raise_for_status()
    df = pd.DataFrame(response.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])

    gaps = df["time"].diff().dropna().ne(pd.Timedelta(hours=1)).sum()
    print(f"rows={len(df)}, {df['time'].min()} -> {df['time'].max()}")
    print(f"missing values={int(df.isna().sum().sum())}, gaps in hourly index={int(gaps)}")
    df.to_csv(OUT, index=False)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
