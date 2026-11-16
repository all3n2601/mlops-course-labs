# MLflow Lab 2, redone as a 24-hour temperature forecast for Boston

This is my version of `Mlflow_Labs/Lab2` (Wine Quality classification). I kept the lab's flow: log experiments, tune hyperparameters, register the model, run batch inference and serve it over REST. I replaced the dataset and the problem, so this is not a copy of the original.

**Task:** given the weather at hour `t`, predict the temperature at hour `t + 24`.

## What I changed from the original lab

| Original Lab 2 | This version |
|---|---|
| Wine Quality CSVs, binary classifier (AUC) | Hourly Boston weather, 2023-2025 (26,304 rows), 24-hour-ahead temperature regression (RMSE, MAE, R2) |
| Random train/validation/test split | Split by time: train 2023-01 to 2024-06, validation 2024-07 to 2024-12, test 2025. The model never trains on data from after the period it is scored on |
| Raw columns | 21 features: current weather, temperature lags (1, 3, 6, 12, 24 h), 24 h rolling temperature mean and rain total, 3 h pressure change, and sine/cosine of local hour and day of year |
| Random forest baseline, XGBoost tuning | A "persistence" baseline (tomorrow = now), a random forest baseline, and `HistGradientBoostingRegressor` tuned with Hyperopt (20 trials, each one a nested MLflow run) |
| Stage transition to Production | `production` alias on the registered model, and it only moves if the new version has a lower validation RMSE (champion check) |
| Batch inference with a Spark UDF on a Delta table | `mlflow.pyfunc.load_model` on a CSV of 24 unseen hours, no Spark needed |
| Metrics only | Also logs a forecast-vs-actual plot and a permutation importance chart as run artifacts |
| Default local store | SQLite backend (`sqlite:///mlflow.db`) so the model registry works locally |

## Data

`fetch_data.py` downloads hourly weather for the Northeastern campus coordinates (42.3398, -71.0892) from the [Open-Meteo historical weather API](https://open-meteo.com) (reanalysis data, see their licence page for terms). The result is saved in `data/boston_hourly_2023_2025.csv` (about 1.5 MB) so the lab does not depend on the API at run time. Times are in UTC. The download had 26,304 rows, no missing values and no gaps in the hourly index (`evidence/fetch_data.txt`).

## Files

- `fetch_data.py` - downloads the weather data
- `train.py` - features, time split, baselines, Hyperopt tuning, artifacts, registration and champion check
- `batch_inference.py` - loads `models:/boston_temp_24h_forecaster@production` and scores 24 unseen hours
- `query_server.py` - sends rows to the served model on port 5001
- `data/` - the weather CSV, the 24 sample rows with their true values, and the batch predictions
- `evidence/` - output of the runs below

## Run it

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python train.py            # data/boston_hourly_2023_2025.csv is already included
python batch_inference.py

# serve (the venv must be active so uvicorn is on PATH)
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
mlflow models serve -m "models:/boston_temp_24h_forecaster@production" -h 127.0.0.1 -p 5001 --env-manager local
# in another terminal
python query_server.py

# optional: re-download the data, or open the UI (neither re-tested after the final run)
python fetch_data.py
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

## What happened when I ran it

Run on macOS with Python 3.12, MLflow 3.18, scikit-learn 1.9. Temperatures are in degrees Celsius.

First `train.py` run (`evidence/train_run1.txt`):

```
train=13104 val=4416 test=8736, features=21, mean temp=11.0 C
persistence: val RMSE = 3.662, test RMSE = 4.430
random forest: val RMSE = 3.520, R2 = 0.861
best params = {'max_iter': 150, 'learning_rate': 0.035, 'max_depth': 9, 'l2_regularization': 1.151}
tuned val RMSE = 3.235, R2 = 0.883
tuned test RMSE = 3.746, MAE = 2.889, R2 = 0.875 (persistence test RMSE 4.430)
version 1 (RMSE 3.2348) is now 'production' (previous: inf)
```

On the 2025 test year the tuned model's RMSE is 3.75 C against 4.43 C for the persistence baseline, so it beats "tomorrow equals now" by about 0.7 C.

Second run (`evidence/train_run2.txt`): same seed, same result, so version 2 tied with version 1 and the alias stayed on version 1:

```
version 2 (RMSE 3.2348) did not beat current 'production' (RMSE 3.2348), alias unchanged
```

REST serving: `/health` returned HTTP 200 and the first 5 forecasts matched the batch run exactly (`evidence/rest_query.txt`).

### One sample day where the model lost

Batch inference scored the 24 hours starting 2025-07-01 00:00 UTC (`evidence/batch_inference.txt`). On this day the model's mean absolute error was **2.54 C and persistence's was 0.60 C**, so persistence won. The weather stayed hot and steady and the model under-predicted it by up to about 4 C in the afternoon, pulling the forecast toward typical July temperatures. I chose this day before seeing any results and kept it. It is 24 hours out of 8,736 test hours, so it shows a failure mode (heat waves and other persistent patterns), not the average performance.

## Notes and limits

- The tuned model only beats persistence on average over the test year. It does not beat it every day.
- The validation period is only the second half of 2024, so tuning sees one summer-to-winter stretch.
- Models are logged with `serialization_format="cloudpickle"`. MLflow 3.x defaults to `skops` for scikit-learn, which refused to save the random forest because of its tree types. Only load models you trained yourself.
- Weather here is reanalysis data for one point, not a station record.
- `mlflow.db` and `mlruns/` are in `.gitignore`, so run `train.py` to recreate them.
- I did not run this on Ubuntu/Docker and I did not capture MLflow UI screenshots.
