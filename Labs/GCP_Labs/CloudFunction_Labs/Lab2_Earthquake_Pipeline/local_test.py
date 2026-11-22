"""Runs the three stages' logic locally (no GCP) with the same pinned library versions."""
import importlib.util
import json
import sys
import types

import pandas as pd


def load(path, name):
    # stub the GCP clients so the modules import offline
    for mod in ["google.cloud.pubsub_v1", "google.cloud.storage"]:
        sys.modules.setdefault(mod, types.ModuleType(mod))
    sys.modules["google.cloud.storage"].Client = lambda *a, **k: None
    sys.modules["google.cloud.pubsub_v1"].PublisherClient = lambda *a, **k: None
    import google.cloud
    google.cloud.storage = sys.modules["google.cloud.storage"]
    google.cloud.pubsub_v1 = sys.modules["google.cloud.pubsub_v1"]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


process = load("src/process_data/main.py", "process_data_main")
train = load("src/train_model/main.py", "train_model_main")
serve = load("src/predict/main.py", "predict_main")

raw = pd.read_csv("data/usgs_earthquakes_2025.csv")
df, report = process.clean(raw)
print(json.dumps(report, indent=2))
# same round trip as the cloud: write the cleaned CSV as text, read it back in the next stage
csv_text = df.assign(time=df["time"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")).to_csv(index=False)
import io
df = pd.read_csv(io.StringIO(csv_text))
model, metrics = train.fit_and_select(df)
print(json.dumps(metrics, indent=2))
print(json.dumps(serve.score(model, [
    {"latitude": 52.9, "longitude": 160.0, "depth": 20, "gap": 80, "dmin": 1.1, "rms": 0.9, "nst": 150},
    {"latitude": 34.0, "longitude": -117.4, "depth": 8, "rms": 0.2},
]), indent=2))
# bad input checks for clean()
try:
    process.clean(raw.drop(columns=["rms"]))
except ValueError as e:
    print("missing-column check OK:", e)
