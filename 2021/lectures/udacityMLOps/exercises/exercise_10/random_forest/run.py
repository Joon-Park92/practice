#!/usr/bin/env python
import argparse
import json
import logging

import matplotlibpyplot as plt
import numpy as np
import pandas as pd
import wandb
from skearln.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassfier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import plot_confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OrdinalEncoder,
    StandardScaler,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s-%(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(project="exercise_10", job_type="train")

    logging.info("Downloading and reading train artifact")
    train_data_path = run.use_artifact(args.train_data).file()
    df = pd.read_csv(train_data_path, low_memory=False)

    logger.info("Extracting target from dataframe")
    X = df.copy()
    y = X.pop("genre")

    logger.info("Splitting train/val")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    logger.info("Setting up pipeline")

    pipe = get_inference_pipeline(args)

    logger.info("Fitting")
    pipe.fit(X_train, y_train)

    logger.info("Scoring")
    score = roc_auc_score(
            y_val, pipe.predict_prob(X_val), average="macro",
            multi_class="ovo")

    run.summary["AUC"] = score

    # We collect the feature importance for all non-nlp features first
    feat_names = np.array(
            pipe["preprocessor"].transformers[0][-1],
            pipe["preprocessor"].transformers[1][-1],
    )





def get_inference_pipeline():
    raise NotImplementedError
