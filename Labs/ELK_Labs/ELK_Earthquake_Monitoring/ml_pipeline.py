"""Train a classifier on early-2025 earthquakes, then replay the rest of the year week by
week and log model quality and data drift to plain-text logs for Logstash.

Task: flag whether an event is magnitude 4.5 or higher from its location, depth and the
quality of the seismic readings (stations, gap, distance, rms). It is a toy task: the point
of the lab is the logging, ELK pipeline and drift monitoring, not seismology.

Split (fixed before looking at results):
  train      2025-01-01 .. 2025-05-31
  validation 2025-06-01 .. 2025-06-30   (picks the best of several configs)
  replay     2025-07-01 .. 2025-12-31   (weekly batches)
Drift rule: a feature drifts in a batch when its two-sample KS statistic against the
training data is at least KS_THRESHOLD.
"""
import logging
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = "data/usgs_earthquakes_2025.csv"
FEATURES = ["latitude", "longitude", "depth", "gap", "dmin", "rms", "nst"]
DRIFT_FEATURES = ["latitude", "longitude", "depth", "gap", "dmin", "rms", "nst"]
KS_THRESHOLD = 0.20
MAG_CUTOFF = 4.5
SEED = 42


def make_logger(name, path):
    handler = logging.FileHandler(path, mode="w")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s",
                                  datefmt="%Y-%m-%dT%H:%M:%SZ")
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger


def rates(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "val_f1": f1_score(y_true, y_pred),
        "val_precision": precision_score(y_true, y_pred),
        "val_recall": recall_score(y_true, y_pred),
        "val_fpr": fp / (fp + tn),
        "val_fnr": fn / (fn + tp),
    }


def main():
    os.makedirs("logs", exist_ok=True)
    train_log = make_logger("training", "logs/training.log")
    drift_log = make_logger("drift", "logs/drift.log")

    df = pd.read_csv(DATA, parse_dates=["time"])
    df["time"] = df["time"].dt.tz_localize(None)
    df["target"] = (df["mag"] >= MAG_CUTOFF).astype(int)

    train = df[df["time"] < "2025-06-01"]
    val = df[(df["time"] >= "2025-06-01") & (df["time"] < "2025-07-01")]
    replay = df[df["time"] >= "2025-07-01"].copy()
    medians = train[FEATURES].median()  # fill missing readings with training medians
    X_train, y_train = train[FEATURES].fillna(medians), train["target"]
    X_val, y_val = val[FEATURES].fillna(medians), val["target"]

    train_log.info(f"Starting training | train_rows: {len(train)} | val_rows: {len(val)} | "
                   f"replay_rows: {len(replay)} | share_m45: {y_train.mean():.3f}")

    configs = [
        ("logistic_regression", make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500))),
        ("logistic_regression", make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500))),
        ("gradient_boosting", GradientBoostingClassifier(n_estimators=50, max_depth=2, random_state=SEED)),
        ("gradient_boosting", GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=SEED)),
        ("gradient_boosting", GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                                         random_state=SEED)),
        ("gradient_boosting", GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=SEED)),
    ]
    best, best_f1 = None, -1.0
    for run, (name, model) in enumerate(configs, start=1):
        model.fit(X_train, y_train)
        metrics = rates(y_val, model.predict(X_val))
        train_log.info(f"Training run {run} finished | model: {name} | "
                       + " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))
        if metrics["val_f1"] > best_f1:
            best, best_f1, best_run = model, metrics["val_f1"], run
    train_log.info(f"Selected run {best_run} as the production model | val_f1: {best_f1:.4f}")

    # Replay: weekly batches starting on Monday
    replay["week"] = replay["time"].dt.normalize() - pd.to_timedelta(replay["time"].dt.weekday, unit="D")
    drift_log.info(f"Replay started | ks_threshold: {KS_THRESHOLD}")
    for batch, (week, chunk) in enumerate(replay.groupby("week"), start=1):
        week_str = week.strftime("%Y-%m-%d")
        X_batch = chunk[FEATURES].fillna(medians)
        f1 = f1_score(chunk["target"], best.predict(X_batch))
        drift_log.info(f"Batch {batch} week {week_str} | events: {len(chunk)} | f1: {f1:.4f} | "
                       f"mean_mag: {chunk['mag'].mean():.3f} | share_m45: {chunk['target'].mean():.3f}")
        for feature in DRIFT_FEATURES:
            stat, p = ks_2samp(chunk[feature].dropna(), train[feature].dropna())
            if stat >= KS_THRESHOLD:
                drift_log.warning(f"Data drift detected in batch {batch} week {week_str} | "
                                  f"feature: {feature} | ks_stat: {stat:.4f} | p_value: {p:.6f}")
    drift_log.info("Replay finished")


if __name__ == "__main__":
    main()
