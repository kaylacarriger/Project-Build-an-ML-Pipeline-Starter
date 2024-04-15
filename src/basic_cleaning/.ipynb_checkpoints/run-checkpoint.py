#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd
import os

# DO NOT MODIFY
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# DO NOT MODIFY
def go(args):
    
    logger.info('Starting wandb run.')
    run = wandb.init(
        project = 'nyc_airbnb',
        group = 'basic_cleaning',
        job_type="basic_cleaning" 
    )
    run.config.update(args)
    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    logger.info('Fetching raw dataset.')
    local_path = wandb.use_artifact('sample.csv:latest').file()
    df = pd.read_csv(local_path)
    
    # EDA with arguments passed into the step
    logger.info('Cleaning data.')
    idx = df['price'].between(float(args.min_price), float(args.max_price))
    df = df[idx].copy()
    df['last_review'] = pd.to_datetime(df['last_review'])
    # TODO: add code to fix the issue happened when testing the model
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()
    
def basic_cleaning(args):
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    logger.info("Downloading artifact")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    # Read artifact
    data_frame = pd.read_csv(artifact_path)

    # Dropping outliers
    logger.info("Dropping outlier")
    min_price = args.min_price
    max_price = args.max_price
    idx = data_frame['price'].between(min_price, max_price)
    data_frame = data_frame[idx].copy()

    # Convert last_review to datetime
    logger.info("Convert last_review columns to date")
    data_frame['last_review'] = pd.to_datetime(data_frame['last_review'])

    # Drop longitude and latitude outliers
    idx = data_frame['longitude'].between(-74.25, - \
                                          73.50) & data_frame['latitude'].between(40.5, 41.2)
    data_frame = data_frame[idx].copy()

    filename = args.output_artifact
    data_frame.to_csv(filename, index=False)

    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(filename)

    logger.info("Logging artifact")
    run.log_artifact(artifact)

    os.remove(filename)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Name for the input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type for the artifact",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description for the artifact",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum value for price",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum value for price",
        required=True
    )

    args = parser.parse_args()

    basic_cleaning(args)