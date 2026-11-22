"""Stage 3. HTTP function (authenticated) that serves the trained model.

POST {"events": [{"latitude":..., "longitude":..., "depth":..., "gap":..., "dmin":..., "rms":..., "nst":...}]}
returns, for each event, the probability of magnitude 4.5 or higher and a yes/no flag.
Missing readings (gap, dmin, rms, nst) may be left out or null.
"""
import json
import os

import functions_framework
import joblib
import pandas as pd
from google.cloud import storage

FEATURES = ["latitude", "longitude", "depth", "gap", "dmin", "rms", "nst"]
REQUIRED = ["latitude", "longitude", "depth"]
_model = None


def load_model():
    global _model
    if _model is None:
        bucket = storage.Client().bucket(os.environ["BUCKET"])
        bucket.blob("models/model.joblib").download_to_filename("/tmp/model.joblib")
        _model = joblib.load("/tmp/model.joblib")
        print("model loaded from Cloud Storage")
    return _model


def score(model, events):
    """Pure function, tested locally."""
    frame = pd.DataFrame(events).reindex(columns=FEATURES).astype(float)
    proba = model.predict_proba(frame)[:, 1]
    return [{"probability_m45_or_higher": round(float(p), 4), "predicted_m45_or_higher": bool(p >= 0.5)}
            for p in proba]


@functions_framework.http
def predict(request):
    body = request.get_json(silent=True) or {}
    events = body.get("events")
    if not isinstance(events, list) or not events:
        return (json.dumps({"error": "send {\"events\": [ {...}, ... ]}"}), 400,
                {"Content-Type": "application/json"})
    for i, event in enumerate(events):
        missing = [f for f in REQUIRED if event.get(f) is None]
        if missing:
            return (json.dumps({"error": f"event {i} is missing {missing}"}), 400,
                    {"Content-Type": "application/json"})
    results = score(load_model(), events)
    return json.dumps({"predictions": results}), 200, {"Content-Type": "application/json"}
