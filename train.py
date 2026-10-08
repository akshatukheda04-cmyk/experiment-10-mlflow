"""Train and log one Iris Random Forest run for Experiment 10."""

import argparse
import shutil
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"
TRACKING_URI = (HERE / "mlruns").as_uri()
EXPERIMENT_NAME = "Iris_Classification"
INPUT_EXAMPLE = np.array([[5.1, 3.5, 1.4, 0.2]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=5)
    args = parser.parse_args()
    if args.n_estimators < 1 or args.max_depth < 1:
        parser.error("n-estimators and max-depth must be positive integers")

    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
    )
    model.fit(x_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(x_test))

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(
        run_name=f"random-forest-{args.n_estimators}-{args.max_depth}"
    ) as run:
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("dataset", "scikit-learn Iris")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(
            sk_model=model,
            name="random_forest_model",
            input_example=INPUT_EXAMPLE,
        )
        print(f"Run ID: {run.info.run_id}")

    # Keep the latest trained model available for the serving step.
    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)
    mlflow.sklearn.save_model(
        sk_model=model,
        path=str(MODEL_DIR),
        input_example=INPUT_EXAMPLE,
    )
    print(f"Model Accuracy: {accuracy:.3f}")
    print(f"Saved model: {MODEL_DIR}")


if __name__ == "__main__":
    main()
