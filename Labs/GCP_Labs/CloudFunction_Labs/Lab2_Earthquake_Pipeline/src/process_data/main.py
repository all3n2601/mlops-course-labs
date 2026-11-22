"""Stage 1. Triggered when a CSV lands in gs://BUCKET/raw/.

Validates and cleans the USGS earthquake file, writes processed/clean.csv and a
data-quality report, then publishes a message that starts the training stage.
"""
import io
import json
import os

import functions_framework
import pandas as pd
from google.cloud import pubsub_v1, storage

FEATURES = ["latitude", "longitude", "depth", "gap", "dmin", "rms", "nst"]
REQUIRED = ["time", "mag"] + FEATURES
CORE = ["time", "mag", "latitude", "longitude", "depth"]  # a row without these is unusable

storage_client = storage.Client()
publisher = pubsub_v1.PublisherClient()


def clean(df):
    """Return (clean dataframe, quality report). Pure function, tested locally."""
    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        raise ValueError(f"missing columns: {missing_cols}")
    report = {"rows_in": int(len(df))}

    df = df[REQUIRED].copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    for col in REQUIRED[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=CORE)
    report["dropped_missing_core_field"] = int(before - len(df))

    before = len(df)
    df = df[df["latitude"].between(-90, 90) & df["longitude"].between(-180, 180)
            & df["mag"].between(-2, 10)]
    report["dropped_out_of_range"] = int(before - len(df))

    before = len(df)
    df = df.drop_duplicates(subset=["time", "latitude", "longitude", "mag"])
    report["dropped_duplicates"] = int(before - len(df))

    df = df.sort_values("time").reset_index(drop=True)
    report.update({
        "rows_out": int(len(df)),
        "time_min": df["time"].min().isoformat(),
        "time_max": df["time"].max().isoformat(),
        "share_m45_or_higher": round(float((df["mag"] >= 4.5).mean()), 4),
        "missing_by_feature": {c: int(df[c].isna().sum()) for c in FEATURES},
    })
    return df, report


@functions_framework.cloud_event
def process_data(cloud_event):
    data = cloud_event.data
    bucket_name, name = data["bucket"], data["name"]
    if not (name.startswith("raw/") and name.endswith(".csv")):
        print(f"ignoring {name}: not a raw/*.csv upload")
        return

    bucket = storage_client.bucket(bucket_name)
    raw = pd.read_csv(io.StringIO(bucket.blob(name).download_as_text()))
    df, report = clean(raw)
    report["source"] = f"gs://{bucket_name}/{name}"

    out = df.assign(time=df["time"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))  # one fixed time format
    bucket.blob("processed/clean.csv").upload_from_string(out.to_csv(index=False), "text/csv")
    bucket.blob("processed/quality_report.json").upload_from_string(
        json.dumps(report, indent=2), "application/json")
    print("quality report:", json.dumps(report))

    topic = publisher.topic_path(os.environ["GCP_PROJECT"], os.environ["TRAIN_TOPIC"])
    message = {"bucket": bucket_name, "file": "processed/clean.csv", "rows": report["rows_out"]}
    publisher.publish(topic, json.dumps(message).encode()).result(timeout=30)
    print(f"published {message} to {topic}")
