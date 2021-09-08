import random

import mlflow


if __name__ == "__main__":

    mlflow.set_tracking_uri("http://localhost:5000")

    # Create or Get an experiment and log two runs under it
    experiment = mlflow.get_experiment_by_name("Pycon2021")

    if not experiment:
        experiment_id = mlflow.create_experiment("Pycon2021")

    else:
        experiment_id = experiment.experiment_id

    # Run Experiments
    for exp in range(1, 10):
        with mlflow.start_run(experiment_id=experiment_id):
            for i in range(10):
                step = i
                value = random.random() +  (step / 5)
                mlflow.log_metric("m", value, step=step)

            mlflow.set_tag("s.release", "1.1.0-RC")
            mlflow.log_param("learning_rate", exp/10)
            mlflow.log_param("num_layer", exp)
