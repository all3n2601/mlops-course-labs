"""Original regression experiment using the TensorBoard Lab 4 HParams workflow."""
import hashlib
import itertools
import json
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorboard.plugins.hparams import api as hp

ROOT = Path(__file__).resolve().parent
FEATURES = [f"X{i}" for i in range(1, 9)]
SEED = 42


def score(actual, predicted):
    return {"mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
            "r2": float(r2_score(actual, predicted))}


def dataset(x, y, training=False):
    ds = tf.data.Dataset.from_tensor_slices((x.astype("float32"), y.astype("float32")))
    if training:
        ds = ds.shuffle(len(x), seed=SEED, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    return ds.batch(32).with_options(options)


def run():
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    logs = ROOT / "logs"
    # Preserve the evidence for an existing experiment instead of mixing runs.
    if logs.exists() and any(logs.iterdir()):
        raise FileExistsError("logs/ contains a previous run. Move it aside before a new experiment.")
    raw = ROOT / "data/ENB2012_data.xlsx"
    df = pd.read_excel(raw).dropna(how="all").loc[:, FEATURES + ["Y1", "Y2"]]
    assert df.shape == (768, 10) and not df.isna().any().any()
    assert np.isfinite(df.to_numpy()).all()
    # X1 identifies each of the 12 building geometries in this dataset.
    groups = df.X1.to_numpy()
    trainval, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
                          .split(df, groups=groups))
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
                .split(df.iloc[trainval], groups=groups[trainval]))
    train, val = trainval[a], trainval[b]
    splits = {"train": train.tolist(), "validation": val.tolist(), "test": test.tolist()}
    (results / "splits.json").write_text(json.dumps(splits, indent=2))
    numeric = ["X1", "X2", "X3", "X4", "X5", "X7"]
    pre = ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("categorical", OneHotEncoder(categories=[[2, 3, 4, 5], [0, 1, 2, 3, 4, 5]],
                                      sparse_output=False, handle_unknown="error"), ["X6", "X8"]),
    ])
    x_train = pre.fit_transform(df.iloc[train][FEATURES]).astype("float32")
    x_val = pre.transform(df.iloc[val][FEATURES]).astype("float32")
    # Only training labels determine target scaling.
    y_mean, y_std = float(df.iloc[train].Y1.mean()), float(df.iloc[train].Y1.std(ddof=0))
    y_train = ((df.iloc[train].Y1.to_numpy() - y_mean) / y_std).astype("float32")
    y_val = ((df.iloc[val].Y1.to_numpy() - y_mean) / y_std).astype("float32")
    preprocessing = {"numeric_columns": numeric,
                     "mean": pre.named_transformers_["numeric"].mean_.tolist(),
                     "scale": pre.named_transformers_["numeric"].scale_.tolist(),
                     "categorical_columns": ["X6", "X8"],
                     "categories": [[2, 3, 4, 5], [0, 1, 2, 3, 4, 5]],
                     "target_mean": y_mean, "target_std": y_std}
    (results / "preprocessing.json").write_text(json.dumps(preprocessing, indent=2))
    hp_units = hp.HParam("units", hp.Discrete([32, 64]))
    hp_dropout = hp.HParam("dropout", hp.Discrete([0.0, 0.1]))
    hp_lr = hp.HParam("learning_rate", hp.Discrete([0.001, 0.01]))
    with tf.summary.create_file_writer(str(logs)).as_default():
        hp.hparams_config(hparams=[hp_units, hp_dropout, hp_lr],
                          metrics=[hp.Metric("validation_mae", display_name="Validation MAE")])
    trials, histories = [], {}
    best_mae = float("inf")
    for i, (units, dropout, lr) in enumerate(itertools.product([32, 64], [0.0, 0.1], [0.001, 0.01])):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED)
        run_name = f"trial-{i:02d}"
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(x_train.shape[1],)),
            tf.keras.layers.Dense(units, activation="relu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(units // 2, activation="relu"),
            tf.keras.layers.Dense(1),
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse",
                      metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")])
        history = model.fit(dataset(x_train, y_train, True), validation_data=dataset(x_val, y_val),
                            epochs=150, verbose=0, callbacks=[
                                tf.keras.callbacks.TensorBoard(log_dir=str(logs / run_name), histogram_freq=0),
                                tf.keras.callbacks.EarlyStopping(monitor="val_mae", patience=20,
                                                                 restore_best_weights=True),
                            ])
        pred = model(x_val, training=False).numpy().ravel() * y_std + y_mean
        metrics = score(df.iloc[val].Y1, pred)
        row = {"run": run_name, "units": units, "dropout": dropout, "learning_rate": lr,
               "epochs": len(history.history["loss"]),
               "best_epoch": int(np.argmin(history.history["val_mae"]) + 1),
               **{"validation_" + k: v for k, v in metrics.items()}}
        trials.append(row)
        histories[run_name] = history.history
        with tf.summary.create_file_writer(str(logs / run_name)).as_default():
            hp.hparams({hp_units: units, hp_dropout: dropout, hp_lr: lr})
            tf.summary.scalar("validation_mae", metrics["mae"], step=row["best_epoch"])
        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_row = row
            model.save(results / "best_model.keras")
        print(json.dumps(row), flush=True)
    pd.DataFrame(trials).to_csv(results / "trials.csv", index=False)
    (results / "histories.json").write_text(json.dumps(histories, indent=2))
    # Unlock the test set only after all selection decisions are complete.
    selected = tf.keras.models.load_model(results / "best_model.keras")
    x_test = pre.transform(df.iloc[test][FEATURES]).astype("float32")
    actual = df.iloc[test].Y1.to_numpy()
    predicted = selected(x_test, training=False).numpy().ravel() * y_std + y_mean
    baseline = np.full(len(test), y_mean)
    predictions = pd.DataFrame({"row": test, "actual": actual, "predicted": predicted,
                                "baseline": baseline})
    predictions.to_csv(results / "test_predictions.csv", index=False)
    summary = {"selected": best_row, "test": score(actual, predicted),
               "mean_baseline_test": score(actual, baseline),
               "split_rows": {k: len(v) for k, v in splits.items()},
               "split_geometries": {k: sorted(set(groups[v].tolist())) for k, v in splits.items()},
               "seed": SEED, "data_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
               "environment": {"python": platform.python_version(), "platform": platform.platform(),
                               "tensorflow": tf.__version__}}
    (results / "summary.json").write_text(json.dumps(summary, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for row in trials:
        h = histories[row["run"]]
        axes[0].plot(np.arange(1, len(h["val_mae"]) + 1), np.array(h["val_mae"]) * y_std,
                     label=row["run"], alpha=0.8)
    axes[0].set(xlabel="Epoch", ylabel="Validation MAE (target units)", title="Eight tuning experiments")
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].scatter(actual, predicted, alpha=0.65, s=22)
    lim = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    axes[1].plot(lim, lim, "k--", label="Perfect prediction")
    axes[1].set(xlabel="Actual heating load", ylabel="Predicted heating load", title="Unseen building geometries: test set")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(results / "evaluation.png", dpi=170)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
