"""Batch inference with the registered production model (replaces the Spark UDF step)."""
import mlflow
import mlflow.pyfunc
import pandas as pd

from train import ALIAS, MODEL_NAME, TRACKING_URI

mlflow.set_tracking_uri(TRACKING_URI)

model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
new_data = pd.read_csv("data/new_samples.csv")
info = pd.read_csv("data/new_samples_actual.csv")

result = info.copy()
result["persistence"] = new_data["temperature_2m"].values
result["predicted"] = model.predict(new_data)
result["model_error"] = (result["predicted"] - result["actual"]).abs()
result["persistence_error"] = (result["persistence"] - result["actual"]).abs()
result.to_csv("data/batch_predictions.csv", index=False)

print(result.round(2).to_string())
print(f"mean absolute error over these {len(result)} hours: "
      f"model {result['model_error'].mean():.2f} C, persistence {result['persistence_error'].mean():.2f} C")
