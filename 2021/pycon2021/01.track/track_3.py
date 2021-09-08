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

    # Search all runs in experiment_id
    df = mlflow.search_runs([experiment_id], order_by=["metrics.m DESC"])
    print(df[["metrics.m", "tags.s.release", "run_id"]])
    print("--")

    # Search the experiment_id using a filter_string with tag
    # that has a case insensitive pattern
    filter_string = "tags.s.release ILIKE '%rc%'"
    df = mlflow.search_runs([experiment_id], filter_string=filter_string)
    print(df[["metrics.m", "tags.s.release", "run_id"]])
