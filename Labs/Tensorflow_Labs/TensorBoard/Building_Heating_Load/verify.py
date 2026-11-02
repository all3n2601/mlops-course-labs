"""Check split isolation, metrics, exported-model inference, and HParams logs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from experiment import FEATURES, score

root = Path(__file__).resolve().parent
results = root / "results"
df = pd.read_excel(root / "data/ENB2012_data.xlsx").dropna(how="all")
splits = json.loads((results / "splits.json").read_text())
summary = json.loads((results / "summary.json").read_text())
train, val, test = (set(splits[k]) for k in ["train", "validation", "test"])
assert not (train & val or train & test or val & test)
assert train | val | test == set(range(768))
tg, vg, eg = (set(df.iloc[sorted(v)].X1) for v in [train, val, test])
assert not (tg & vg or tg & eg or vg & eg)
assert len(set(df.X1)) == 12
assert df.groupby("X1")[["X2", "X3", "X4", "X5"]].nunique().max().max() == 1
print("PASS: all 768 rows assigned once; 12 geometries isolated across splits")
p = json.loads((results / "preprocessing.json").read_text())
np.testing.assert_allclose(df.iloc[splits["train"]][p["numeric_columns"]].mean(), p["mean"])
assert np.isclose(df.iloc[splits["train"]].Y1.mean(), p["target_mean"])
print("PASS: preprocessing and target scaling fitted using training rows")
trials = pd.read_csv(results / "trials.csv")
assert len(trials) == 8
assert trials.loc[trials.validation_mae.idxmin(), "run"] == summary["selected"]["run"]
assert all("test" not in column for column in trials.columns)
print("PASS: eight trials; selection uses minimum validation MAE")
frame = df.iloc[splits["test"]]
numeric = (frame[p["numeric_columns"]].to_numpy() - p["mean"]) / p["scale"]
categorical = [np.column_stack([frame[col].to_numpy() == v for v in values])
               for col, values in zip(p["categorical_columns"], p["categories"])]
x = np.column_stack([numeric, *categorical]).astype("float32")
model = tf.keras.models.load_model(results / "best_model.keras")
pred = model(x, training=False).numpy().ravel() * p["target_std"] + p["target_mean"]
recorded = pd.read_csv(results / "test_predictions.csv")
np.testing.assert_allclose(pred, recorded.predicted, rtol=1e-5, atol=1e-5)
for key, value in score(frame.Y1, pred).items():
    assert np.isclose(value, summary["test"][key], rtol=1e-5)
print("PASS: saved model reproduces recorded predictions and test metrics")
for run in trials.run:
    accumulator = EventAccumulator(str(root / "logs" / run)).Reload()
    assert "_hparams_/session_start_info" in accumulator.Tags()["tensors"]
    assert "validation_mae" in accumulator.Tags()["tensors"]
    value = tf.make_ndarray(accumulator.Tensors("validation_mae")[0].tensor_proto).item()
    expected = float(trials.loc[trials.run == run, "validation_mae"].iloc[0])
    assert np.isclose(value, expected)
print("PASS: eight TensorBoard HParams sessions contain the recorded validation metrics")
print("All verification checks passed.")
