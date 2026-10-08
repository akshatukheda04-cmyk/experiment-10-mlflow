# Experiment 10: Iris classification with MLflow

This is a small, standalone project for demonstrating the MLflow model lifecycle with the Iris dataset. It trains three Random Forest configurations, records their settings and accuracy, saves the latest model, and exposes a local prediction endpoint.

## Run in VS Code on Windows

Open this folder in VS Code. In its PowerShell terminal, run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Train the three configurations one at a time:

```powershell
.\.venv\Scripts\python.exe train.py --n-estimators 100 --max-depth 5
.\.venv\Scripts\python.exe train.py --n-estimators 50 --max-depth 3
.\.venv\Scripts\python.exe train.py --n-estimators 200 --max-depth 10
```

## Show the MLflow results

In a second VS Code terminal, from this folder, start the dashboard:

```powershell
.\.venv\Scripts\mlflow.exe ui --backend-store-uri .\mlruns --host 127.0.0.1 --port 5000
```

Open <http://127.0.0.1:5000> and choose the `Iris_Classification` experiment to compare the three runs, their parameters, and accuracy.

## Demonstrate a prediction

Stop the dashboard with Ctrl+C. Start the model server:

```powershell
$env:PATH = (Resolve-Path .\.venv\Scripts).Path + ';' + $env:PATH
.\.venv\Scripts\mlflow.exe models serve --model-uri .\model --port 5001 --host 127.0.0.1 --env-manager local
```

Keep that terminal open. In another terminal, from this folder, send the sample request:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5001/invocations -ContentType 'application/json' -Body (Get-Content .\input.json -Raw)
```

The response should contain a prediction of `0`, `1`, or `2` for the Iris flower class.

Generated `mlruns/`, `model/`, and `.venv/` files are local demo data and are excluded from Git.
