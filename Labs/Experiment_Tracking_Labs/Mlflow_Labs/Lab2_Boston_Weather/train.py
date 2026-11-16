"""MLflow Lab 2, redone as a 24-hour temperature forecast for Boston.

Steps: load hourly weather, build lag features, split by time (not randomly), log a
persistence baseline and a random forest baseline, tune a gradient boosting model with
hyperopt (nested runs), then register it and move the `production` alias only if it
beats the current production version (lower validation RMSE).
"""
import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from mlflow import MlflowClient
from mlflow.models.signature import infer_signature
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "boston-temperature-24h"
MODEL_NAME = "boston_temp_24h_forecaster"
ALIAS = "production"
DATA = "data/boston_hourly_2023_2025.csv"
HORIZON = 24  # predict the temperature this many hours after the row's timestamp
SEED = 123


def build_dataset():
    df = pd.read_csv(DATA, parse_dates=["time"])
    temp = df["temperature_2m"]
    X = df.drop(columns="time").copy()

    for lag in (1, 3, 6, 12, 24):
        X[f"temp_lag_{lag}h"] = temp.shift(lag)
    X["temp_mean_24h"] = temp.rolling(24).mean()
    X["precip_sum_24h"] = df["precipitation"].rolling(24).sum()
    X["pressure_change_3h"] = df["pressure_msl"] - df["pressure_msl"].shift(3)

    # The daily cycle follows local time, so convert from UTC before taking the hour
    local = df["time"].dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    X["hour_sin"] = np.sin(2 * np.pi * local.dt.hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * local.dt.hour / 24)
    X["doy_sin"] = np.sin(2 * np.pi * local.dt.dayofyear / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * local.dt.dayofyear / 365.25)

    y = temp.shift(-HORIZON).rename("temp_in_24h")
    keep = X.notna().all(axis=1) & y.notna()
    return X[keep], y[keep], df.loc[keep, "time"]


def regression_metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def log_eval(y_pred, y, prefix):
    metrics = regression_metrics(y, y_pred)
    mlflow.log_metrics({f"{prefix}_{k}": v for k, v in metrics.items()})
    return metrics


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()

    X, y, times = build_dataset()
    # Split by time so the model never sees the future: 2023-01..2024-06 train,
    # 2024-07..2024-12 validation, 2025 test.
    is_train = times < "2024-07-01"
    is_val = (times >= "2024-07-01") & (times < "2025-01-01")
    is_test = times >= "2025-01-01"
    X_train, y_train = X[is_train], y[is_train]
    X_val, y_val = X[is_val], y[is_val]
    X_test, y_test = X[is_test], y[is_test]
    print(f"train={len(X_train)} val={len(X_val)} test={len(X_test)}, "
          f"features={X.shape[1]}, mean temp={y.mean():.1f} C")

    # Keep one unseen day (24 consecutive test hours) for the batch and REST scripts.
    day = (times[is_test] >= "2025-07-01") & (times[is_test] < "2025-07-02")
    sample_idx = X_test[day.values].index
    X.loc[sample_idx].to_csv("data/new_samples.csv", index=False)
    pd.DataFrame({
        "time": times.loc[sample_idx].values,
        "target_time": times.loc[sample_idx].values + np.timedelta64(HORIZON, "h"),
        "actual": y.loc[sample_idx].values,
    }).to_csv("data/new_samples_actual.csv", index=False)

    # Baseline 1: persistence, "the temperature in 24h equals the temperature now"
    with mlflow.start_run(run_name="baseline_persistence"):
        mlflow.log_params({"model": "persistence", "rule": "temp_in_24h = temperature_2m_now"})
        pers_val = log_eval(X_val["temperature_2m"], y_val, "val")
        pers_test = log_eval(X_test["temperature_2m"], y_test, "test")
        print(f"persistence: val RMSE = {pers_val['rmse']:.3f}, test RMSE = {pers_test['rmse']:.3f}")

    # Baseline 2: random forest with default-ish settings
    with mlflow.start_run(run_name="baseline_random_forest") as run:
        rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=3, n_jobs=-1, random_state=SEED)
        rf.fit(X_train, y_train)
        mlflow.log_params({"model": "RandomForest", "n_estimators": 100, "min_samples_leaf": 3})
        base = log_eval(rf.predict(X_val), y_val, "val")
        signature = infer_signature(X_train, rf.predict(X_train))
        mlflow.sklearn.log_model(rf, name="model", signature=signature,
                                 input_example=X_train.head(3), serialization_format="cloudpickle")
        print(f"random forest: val RMSE = {base['rmse']:.3f}, R2 = {base['r2']:.3f} (run {run.info.run_id})")

    # Tuning: histogram gradient boosting with hyperopt, every trial is a nested run
    space = {
        "max_iter": scope.int(hp.quniform("max_iter", 100, 600, 50)),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.02), np.log(0.3)),
        "max_depth": scope.int(hp.quniform("max_depth", 3, 12, 1)),
        "l2_regularization": hp.loguniform("l2_regularization", np.log(1e-3), np.log(10)),
    }

    def objective(params):
        with mlflow.start_run(nested=True):
            hgb = HistGradientBoostingRegressor(random_state=SEED, **params)
            hgb.fit(X_train, y_train)
            mlflow.log_params(params)
            val = log_eval(hgb.predict(X_val), y_val, "val")
        return {"loss": val["rmse"], "status": STATUS_OK}

    with mlflow.start_run(run_name="hist_gradient_boosting_tuning"):
        trials = Trials()
        best = fmin(
            objective, space, algo=tpe.suggest, max_evals=20, trials=trials,
            rstate=np.random.default_rng(SEED), show_progressbar=False,
        )
        best_params = {
            "max_iter": int(best["max_iter"]),
            "learning_rate": float(best["learning_rate"]),
            "max_depth": int(best["max_depth"]),
            "l2_regularization": float(best["l2_regularization"]),
        }
        best_model = HistGradientBoostingRegressor(random_state=SEED, **best_params)
        best_model.fit(X_train, y_train)

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        tuned_val = log_eval(best_model.predict(X_val), y_val, "val")
        pred_test = best_model.predict(X_test)
        test = log_eval(pred_test, y_test, "test")
        mlflow.log_metric("test_rmse_gain_vs_persistence", pers_test["rmse"] - test["rmse"])

        # Extra artifacts: two weeks of test forecasts vs reality, permutation importance
        window = (times[is_test] >= "2025-07-01") & (times[is_test] < "2025-07-15")
        t_axis = times[is_test][window.values] + pd.Timedelta(hours=HORIZON)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t_axis, y_test[window.values], label="actual", color="black")
        ax.plot(t_axis, pred_test[window.values], label="model", color="tab:blue")
        ax.plot(t_axis, X_test["temperature_2m"][window.values], label="persistence",
                color="tab:orange", alpha=0.6)
        ax.set(ylabel="Temperature (C)", title="24h-ahead forecast, 1-14 July 2025")
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        mlflow.log_figure(fig, "forecast_vs_actual_july_2025.png")
        plt.close(fig)

        sample = X_val.sample(2000, random_state=SEED)
        perm = permutation_importance(best_model, sample, y_val.loc[sample.index],
                                      n_repeats=5, random_state=SEED)
        imp = pd.Series(perm.importances_mean, index=X.columns).nlargest(12).sort_values()
        fig, ax = plt.subplots(figsize=(7, 5))
        imp.plot.barh(ax=ax)
        ax.set_title("Permutation importance (validation sample)")
        fig.tight_layout()
        mlflow.log_figure(fig, "feature_importance.png")
        plt.close(fig)

        signature = infer_signature(X_train, best_model.predict(X_train))
        info = mlflow.sklearn.log_model(best_model, name="model", signature=signature,
                                        input_example=X_train.head(3),
                                        serialization_format="cloudpickle")
        mlflow.set_tag("stage", "tuned")
        print(f"best params = {best_params}")
        print(f"tuned val RMSE = {tuned_val['rmse']:.3f}, R2 = {tuned_val['r2']:.3f}")
        print(f"tuned test RMSE = {test['rmse']:.3f}, MAE = {test['mae']:.3f}, R2 = {test['r2']:.3f}"
              f" (persistence test RMSE {pers_test['rmse']:.3f})")

    # Champion check: register the tuned model, promote only if it wins
    version = mlflow.register_model(info.model_uri, MODEL_NAME)
    client.set_model_version_tag(MODEL_NAME, version.version, "val_rmse", f"{tuned_val['rmse']:.4f}")

    try:
        current = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        current_rmse = float(current.tags.get("val_rmse", "inf"))
    except Exception:
        current_rmse = float("inf")

    if tuned_val["rmse"] < current_rmse:
        client.set_registered_model_alias(MODEL_NAME, ALIAS, version.version)
        print(f"version {version.version} (RMSE {tuned_val['rmse']:.4f}) is now '{ALIAS}' "
              f"(previous: {current_rmse:.4f})")
    else:
        print(f"version {version.version} (RMSE {tuned_val['rmse']:.4f}) did not beat current "
              f"'{ALIAS}' (RMSE {current_rmse:.4f}), alias unchanged")


if __name__ == "__main__":
    main()
