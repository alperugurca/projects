"""AWS services configuration for LocalStack."""

import os
from typing import Dict, Optional

import boto3
from botocore.config import Config

# LocalStack endpoint configuration
LOCALSTACK_ENDPOINT = "http://localhost:4566"

# Load AWS credentials from environment variables
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "test")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "test")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# S3 configurations
MODEL_BUCKET = "heart-disease-model-artifacts"
DATA_BUCKET = "heart-disease-data"

# SageMaker configurations
SAGEMAKER_MODEL_NAME = "heart-disease-predictor"
SAGEMAKER_ENDPOINT = "heart-disease-endpoint"


def get_aws_session(
    region: Optional[str] = None, endpoint_url: Optional[str] = None
) -> boto3.Session:
    """Create AWS session with proper configuration."""
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=region or AWS_REGION,
    )
    return session


def get_s3_client(session: Optional[boto3.Session] = None) -> boto3.client:
    """Get S3 client with retry configuration."""
    if session is None:
        session = get_aws_session()

    config = Config(retries={"max_attempts": 10, "mode": "standard"})

    try:
        return session.client("s3", config=config)
    except Exception as e:
        print(f"Error creating S3 client: {str(e)}")
        raise


def get_sagemaker_client(session: Optional[boto3.Session] = None) -> boto3.client:
    """Get SageMaker client."""
    if session is None:
        session = get_aws_session()

    try:
        return session.client("sagemaker")
    except Exception as e:
        print(f"Error creating SageMaker client: {str(e)}")
        raise


def get_cloudwatch_client(session: Optional[boto3.Session] = None) -> boto3.client:
    """Get CloudWatch client."""
    if session is None:
        session = get_aws_session()

    try:
        return session.client("cloudwatch")
    except Exception as e:
        print(f"Error creating CloudWatch client: {str(e)}")
        raise


def initialize_s3():
    """Initialize S3 buckets."""
    s3 = get_s3_client()

    # Create buckets if they don't exist
    for bucket in [MODEL_BUCKET, DATA_BUCKET]:
        try:
            s3.head_bucket(Bucket=bucket)
        except:
            s3.create_bucket(Bucket=bucket)
            print(f"Created bucket: {bucket}")


def initialize_sagemaker():
    """Initialize SageMaker resources."""
    sagemaker = get_sagemaker_client()

    # Additional SageMaker setup can be added here
    pass


def initialize_cloudwatch():
    """Initialize CloudWatch resources."""
    cloudwatch = get_cloudwatch_client()

    # Set up model monitoring metrics
    pass


def initialize_aws_services():
    """Initialize all required AWS services."""
    initialize_s3()
    initialize_sagemaker()
    initialize_cloudwatch()
    print("AWS services initialized successfully")
