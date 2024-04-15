#!/usr/bin/env python
"""
This script trains a Random Forest
"""
import argparse
import logging
import os
import tempfile
import shutil
import matplotlib.pyplot as plt

import mlflow
import json

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer

import wandb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline


def delta_date_feature(dates):
    date_cleaned = pd.DataFrame(dates).apply(pd.to_datetime)
    return date_cleaned.apply(lambda d: (d.max() -d).dt.days, axis=0).to_numpy()


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def train_rf(args):
    run = wandb.init(job_type="train_random_forest")
    run.config.update(args)

    # Get the random forest configuration and update W&B
    with open(args.rf_config) as fp:
        rf_config = json.load(fp)
    run.config.update(rf_config)

    # Fix random seed for the random forest
    rf_config['random_state'] = args.random_seed

    trainval_local_path = run.use_artifact(args.trainval_artifact).file()

    X = pd.read_csv(trainval_local_path)
    y = X.pop("price")  # this removes the column "price" from X and puts it into y

    logger.info(f"Minimum price: {y.min()}, Maximum price: {y.max()}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_size, stratify=X[args.stratify_by], random_state=args.random_seed
    )

    logger.info("Preparing sklearn pipeline")

    sk_pipe, processed_features = get_inference_pipeline(rf_config, args.max_tfidf_features)

    logger.info("Fitting")
    sk_pipe.fit(X_train, y_train)

    # Compute r2 and MAE
    logger.info("Scoring")
    r_squared = sk_pipe.score(X_val, y_val)

    y_pred = sk_pipe.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)

    logger.info(f"Score: {r_squared}")
    logger.info(f"MAE: {mae}")

    logger.info("Exporting model")

    if os.path.exists("random_forest_dir"):
        shutil.rmtree("random_forest_dir")

    signature = mlflow.models.infer_signature(X_val[processed_features], y_pred)


    with tempfile.TemporaryDirectory() as temp_dir:

        export_path = os.path.join(temp_dir, "random_forest_dir")

        mlflow.sklearn.save_model(
            sk_pipe,
            export_path,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            signature=signature,
            input_example=X_val.iloc[:2],
        )

        artifact = wandb.Artifact(
            args.output_artifact,
            type="model_export",
            description="Pipeline export for random forest",
            metadata= rf_config
        )
        artifact.add_dir(export_path)

        run.log_artifact(artifact)

        artifact.wait()

    # Plot feature importance
    fig_feat_imp = plot_feature_importance(sk_pipe, processed_features)

    # log the evaluation metrics
    run.summary['r2'] = r_squared
    run.summary['mae'] = mae

    # Upload visualization to W&B
    run.log(
        {
          "feature_importance": wandb.Image(fig_feat_imp),
        }
    )


def plot_feature_importance(pipe, feat_names):
    feat_imp = pipe["random_forest"].feature_importances_[: len(feat_names)-1]

    nlp_importance = sum(pipe["random_forest"].feature_importances_[len(feat_names) - 1:])
    feat_imp = np.append(feat_imp, nlp_importance)
    fig_feat_imp, sub_feat_imp = plt.subplots(figsize=(10, 10))

    sub_feat_imp.bar(range(feat_imp.shape[0]), feat_imp, color="b", align="center")
    _ = sub_feat_imp.set_xticks(range(feat_imp.shape[0]))
    _ = sub_feat_imp.set_xticklabels(np.array(feat_names), rotation=45)
    fig_feat_imp.tight_layout()
    
    return fig_feat_imp





def get_inference_pipeline(rf_config, max_tfidf_features):
    ordinal_categorical = ["room_type"]
    non_ordinal_categorical = ["neighbourhood_group"]
    ordinal_categorical_preproc = OrdinalEncoder()

    non_ordinal_categorical_preproc = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder()
    )

    zero_imputed = [
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
        "longitude",
        "latitude"
    ]
    zero_imputer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0)
    )

    date_imputer = make_pipeline(
        SimpleImputer(strategy='constant', fill_value='2010-01-01'),
        FunctionTransformer(delta_date_feature, check_inverse=False, validate=False)
    )

    reshape = FunctionTransformer(np.reshape, kw_args={"newshape": -1})
    name_tfidf = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=""),
        reshape,
        TfidfVectorizer(
            binary=False,
            max_features=max_tfidf_features,
            stop_words='english'
        ),
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal_cat", ordinal_categorical_preproc, ordinal_categorical),
            ("non_ordinal_cat", non_ordinal_categorical_preproc, non_ordinal_categorical),
            ("impute_zero", zero_imputer, zero_imputed),
            ("transform_date", date_imputer, ["last_review"]),
            ("transform_name", name_tfidf, ["name"])
        ],
        remainder="drop" # This drops the columns that we do not transform
    )

    processed_features = ordinal_categorical + non_ordinal_categorical + zero_imputed + ["last_review", "name"]

    random_Forest = RandomForestRegressor(**rf_config)

    sk_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("random_forest", random_Forest)
        ]
    )

    return sk_pipe, processed_features


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Basic cleaning of dataset")

    parser.add_argument(
        "--trainval_artifact",
        type=str,
        help="Artifact containing the training dataset to be split into train and validation"
    )

    parser.add_argument(
        "--val_size",
        type=float,
        help="Size of the validation split",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Seed for random number generator",
        default=42,
        required=False,
    )

    parser.add_argument(
        "--stratify_by",
        type=str,
        help="Column to use for stratification",
        default="none",
        required=False,
    )

    parser.add_argument(
        "--rf_config",
        help="Random forest configuration",
        default="{}",
    )

    parser.add_argument(
        "--max_tfidf_features",
        help="Maximum number of words for the TFIDF",
        default=10,
        type=int
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output model",
        required=True,
    )

    args = parser.parse_args()

    train_rf(args)