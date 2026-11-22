"""Stage 2. Triggered by a Pub/Sub message from stage 1.

Trains a classifier that flags earthquakes of magnitude 4.5 or higher, picks the
best configuration on a validation month, scores it on the later months it never
saw, and saves models/model.joblib and models/metrics.json to the bucket.
Split by time (like production would see it): train < June, validation June,
test July onward.
"""
import base64
import io
import json
import time

import functions_framework
import joblib
import pandas as pd
from google.cloud import storage
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["latitude", "longitude", "depth", "gap", "dmin", "rms", "nst"]
MAG_CUTOFF = 4.5
SEED = 42

CONFIGS = {
    "logreg_c1": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                       LogisticRegression(C=1.0, max_iter=500)),
    "hgb_depth3": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.1,
                                                         max_iter=150, random_state=SEED),
    "hgb_depth6": lambda: HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05,
                                                         max_iter=300, random_state=SEED),
}


def scores(y, pred):
    return {"f1": round(f1_score(y, pred, zero_division=0), 4),
            "precision": round(precision_score(y, pred, zero_division=0), 4),
            "recall": round(recall_score(y, pred, zero_division=0), 4)}


def fit_and_select(df):
    """Return (best fitted model, metrics dict). Pure function, tested locally."""
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    df["target"] = (df["mag"] >= MAG_CUTOFF).astype(int)
    train = df[df["time"] < "2025-06-01"]
    val = df[(df["time"] >= "2025-06-01") & (df["time"] < "2025-07-01")]
    test = df[df["time"] >= "2025-07-01"]

    runs, fitted, best, best_name, best_f1 = {}, {}, None, None, -1.0
    for name, make in CONFIGS.items():
        model = fitted[name] = make().fit(train[FEATURES], train["target"])
        runs[name] = scores(val["target"], model.predict(val[FEATURES]))
        if runs[name]["f1"] > best_f1:
            best, best_name, best_f1 = model, name, runs[name]["f1"]

    return best, {
        "selected": best_name,
        "validation_runs": runs,
        "test": scores(test["target"], best.predict(test[FEATURES])),
        "test_logreg_baseline": scores(test["target"], fitted["logreg_c1"].predict(test[FEATURES])),
        "rows": {"train": len(train), "validation": len(val), "test": len(test)},
        "share_m45_train": round(float(train["target"].mean()), 4),
        "share_m45_test": round(float(test["target"].mean()), 4),
        "features": FEATURES,
        "magnitude_cutoff": MAG_CUTOFF,
    }


@functions_framework.cloud_event
def train_model(cloud_event):
    message = json.loads(base64.b64decode(cloud_event.data["message"]["data"]))
    client = storage.Client()
    bucket = client.bucket(message["bucket"])
    df = pd.read_csv(io.StringIO(bucket.blob(message["file"]).download_as_text()))

    started = time.time()
    model, metrics = fit_and_select(df)
    metrics["train_seconds"] = round(time.time() - started, 1)
    metrics["trained_at_unix"] = int(time.time())
    metrics["source_file"] = f"gs://{message['bucket']}/{message['file']}"
    print("metrics:", json.dumps(metrics))

    joblib.dump(model, "/tmp/model.joblib")
    bucket.blob("models/model.joblib").upload_from_filename("/tmp/model.joblib")
    bucket.blob("models/metrics.json").upload_from_string(
        json.dumps(metrics, indent=2), "application/json")
    print("saved models/model.joblib and models/metrics.json")
