import random

import mlflow


if __name__ == "__main__":

    mlflow.set_tracking_uri("http://localhost:5000")

    # Create an experiment and log two runs under it
    experiment_id = mlflow.create_experiment("Pycon2021")

    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_metric("m", 1.55)
        mlflow.set_tag("s.release", "1.1.0-RC")
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_artifact("./artifact.png")

    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_metric("m", 2.50)
        mlflow.set_tag("s.release", "1.2.0-GA")
        mlflow.log_param("learning_rate", 0.2)
        mlflow.log_artifact("./artifact.png")
