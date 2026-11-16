"""Send rows to the model served by `mlflow models serve` (port 5001)."""
import sys

import pandas as pd
import requests

url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5001/invocations"
rows = pd.read_csv("data/new_samples.csv").head(5)
info = pd.read_csv("data/new_samples_actual.csv").head(5)

response = requests.post(url, json={"dataframe_split": rows.to_dict(orient="split")}, timeout=30)
response.raise_for_status()
preds = response.json()["predictions"]

for p, (_, row) in zip(preds, info.iterrows()):
    print(f"forecast for {row['target_time']}: {p:.2f} C (actual {row['actual']:.2f} C)")
