import mlflow
import pandas as pd
import os


def mlflow_scenario(scenario, logdir):
    mlflow.set_experiment(scenario)
    strategies = os.listdir(logdir)

    for strategy in strategies:
        # Start an MLflow run
        with mlflow.start_run(run_name=strategy):
            path = os.path.join(
                    logdir,
                    strategy,
                    os.listdir(os.path.join(logdir, strategy))[-1],
                    "metrics.csv")

            # Log the CSV file as an artifact
            mlflow.log_artifact(path)

            # Optionally, log metrics from the CSV
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                mlflow.log_metric("loss", row["loss"], step=row["round"])
                mlflow.log_metric("accuracy", row["accuracy"], step=row["round"])

if __name__ == "__main__":
    os.rmdir('mlruns')

    scenario = "MNIST"
    logdir = os.path.join("logs", "mnist")
    mlflow_scenario(scenario, logdir)

    scenario = "PACS"
    logdir = os.path.join("logs", "pacs")
    mlflow_scenario(scenario, logdir)