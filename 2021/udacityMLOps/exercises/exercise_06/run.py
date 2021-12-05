#!/usr/bin/env python
import argparse
import logging
import os
import tempfile

import pandas as pd
import wandb
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.ini(project="exercise_6", job_type="split_data")

    logger.info("Downloading and reading artifact")
    artifact = run.user_artifact(args.input_artifact)
    artifact_path = artifact.file()

    df = pd.read_csv(artifact_path, low_memory=False)

    # Split first in model_dev/text, then we further divide model_dev in train
    # and validation

    logger.info("Splitting data into train, val and test")
    splits = {}

    splits["train"], splits["test"] = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df[args.startify] if args.startify != "null" else None,
    )

    # Save the artifacts. We use a temporary directory so we do not leav any
    # trace behind

    with tempfile.TemporaryDirectory() as temp_df:

        for split, df in splits.items():

            # Make the artifact name from the provided root plus the name of
            # the split
            artifact_name = f"{args.artifact_root}_{split}.csv"

            # Get the path on disk within the temp directory
            temp_path = os.path.join(temp_dir, artifact_name)

            logger.info(f"Uploading the {split} dataset to {artifact_name}")

            # Save then upload to W&B
            df.to_csv(temp_path)

            artifact = wandb.Artifact(
                name=artifact_name,
                type=args.artifact_type,
                description=f"{split} split of dataset {args.input_artifact}",
            )
            artifact.add_file(temp_path)

            logger.info("logging artifact")
            run.log_artifact(artifact)
            artifact.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
