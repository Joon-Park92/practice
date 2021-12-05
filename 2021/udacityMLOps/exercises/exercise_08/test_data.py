import pytest
import wandb
import pandas as pd
import scipy.stats

# This is global so all etsts are collected under the same
# run

run = wandb.init(project="exercise_8", job_type="data_tests")


@pytest.fixture(scope="session")
def data():

    local_path = run.use_artifact("exercise_6/data_train.csv:latest").file()
    sample1 = pd.read_csv(local_path)

    local_path = run.use_artifact("exercise_6/data_test.csv:latest").file()
    sample2 = pd.read_csv(local_path)

    return sample1, sample2


def test_kolmogorov_smirnov(data):
    sample_1, sample_2 = data

    numerical_columns = [
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms"
    ]

    # Let's decide the Type I error probability ( related to the False Positive Rate)
    alpha = 0.05
    alpha_prime = 1 - (1 - alpha) ** ( 1 / len(numerical_columns))

    for col in numerical_columns:

        ts, p_value = scipy.stats.ks_2samp(
                sample_1[col[,

