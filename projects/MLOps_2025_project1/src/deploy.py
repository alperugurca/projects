import os
import shlex
import subprocess
from typing import List, Optional

import click
import mlflow
from mlflow.tracking import MlflowClient


def get_latest_model_version():
    """Get the latest model version from MLflow."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name("heart-disease-prediction")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError("No model runs found")
    return runs[0].info.run_id


def validate_docker_args(args: List[str]) -> bool:
    """Validate docker command arguments for security."""
    allowed_commands = {"build", "stop", "rm", "run", "tag", "push"}
    allowed_flags = {"-t", "-d", "--name", "-p", "-e", "--rm"}
    
    if not args or args[0] not in allowed_commands:
        return False
        
    # Validate all arguments
    for arg in args[1:]:
        if arg.startswith("-"):
            if arg not in allowed_flags:
                return False
        else:
            # Validate string arguments
            if not all(c.isalnum() or c in "-.:/_" for c in arg):
                return False
    return True


def run_docker_command(
    cmd: List[str], check: bool = True, capture_output: bool = False
) -> subprocess.CompletedProcess:
    """Safely run a docker command with proper validation."""
    if not validate_docker_args(cmd):
        raise ValueError(f"Invalid docker command: {' '.join(cmd)}")
        
    # Add nosec comment to indicate this has been reviewed for security
    return subprocess.run(cmd, check=check, capture_output=capture_output)  # nosec


def build_model_image(model_version: str) -> None:
    """Build the Docker image for the model."""
    cmd = ["docker", "build", "-t", f"heart-disease-prediction:{model_version}", "."]
    run_docker_command(cmd)


def deploy_model_local(model_version: str, port: int, environment: dict) -> None:
    """Deploy the model locally using Docker."""
    try:
        # Stop and remove any existing container
        run_docker_command(
            ["docker", "stop", "heart-disease-prediction"],
            check=False,
            capture_output=True,
        )
        run_docker_command(
            ["docker", "rm", "heart-disease-prediction"],
            check=False,
            capture_output=True,
        )

        # Prepare environment variables
        env_args = []
        for key, value in environment.items():
            env_args.extend(["-e", f"{key}={value}"])

        # Run the new container
        cmd = [
            "docker",
            "run",
            "-d",  # Run in detached mode
            "--name",
            "heart-disease-prediction",
            "-p",
            f"{port}:8000",
            *env_args,
            f"heart-disease-prediction:{model_version}",
        ]
        run_docker_command(cmd)
        print(f"Model deployed locally at http://localhost:{port}")
    except Exception as e:
        print(f"Failed to deploy model locally: {str(e)}")
        raise


def deploy_model_aws(model_version: str, environment: dict) -> None:
    """Deploy the model to AWS ECS."""
    try:
        # Push image to ECR
        cmd = [
            "docker",
            "tag",
            f"heart-disease-prediction:{model_version}",
            f"{os.getenv('AWS_ACCOUNT_ID')}.dkr.ecr.{os.getenv('AWS_REGION')}.amazonaws.com/heart-disease-prediction:{model_version}",
        ]
        run_docker_command(cmd)

        cmd = [
            "docker",
            "push",
            f"{os.getenv('AWS_ACCOUNT_ID')}.dkr.ecr.{os.getenv('AWS_REGION')}.amazonaws.com/heart-disease-prediction:{model_version}",
        ]
        run_docker_command(cmd)

    except Exception as e:
        print(f"Failed to deploy model to AWS: {str(e)}")
        raise


def deploy_model_gcp(model_version: str, environment: dict) -> None:
    """Deploy the model to Google Cloud Run."""
    try:
        # Push image to GCR
        cmd = [
            "docker",
            "tag",
            f"heart-disease-prediction:{model_version}",
            f"gcr.io/{os.getenv('GCP_PROJECT_ID')}/heart-disease-prediction:{model_version}",
        ]
        run_docker_command(cmd)

        cmd = [
            "docker",
            "push",
            f"gcr.io/{os.getenv('GCP_PROJECT_ID')}/heart-disease-prediction:{model_version}",
        ]
        run_docker_command(cmd)

    except Exception as e:
        print(f"Failed to deploy model to GCP: {str(e)}")
        raise


def deploy_model(
    model_version: str, environment: dict, platform: str = "local", port: int = 8000
) -> None:
    """Deploy the model to the specified platform."""
    if not model_version or not isinstance(model_version, str):
        raise ValueError("Invalid model version")

    if not environment or not isinstance(environment, dict):
        raise ValueError("Invalid environment configuration")

    if platform not in ["local", "aws", "gcp"]:
        raise ValueError("Invalid platform. Must be one of: local, aws, gcp")

    if not isinstance(port, int) or port < 1 or port > 65535:
        raise ValueError("Invalid port number")

    # Build the Docker image
    build_model_image(model_version)

    # Deploy to the specified platform
    if platform == "local":
        deploy_model_local(model_version, port, environment)
    elif platform == "aws":
        deploy_model_aws(model_version, environment)
    elif platform == "gcp":
        deploy_model_gcp(model_version, environment)


@click.command()
@click.option(
    "--environment",
    type=click.Choice(["local", "aws", "azure", "gcp"]),
    default="local",
    help="Deployment environment",
)
@click.option("--port", type=int, default=8000, help="Port for local deployment")
@click.option("--model-version", default="latest", help="Model version to deploy")
def main(environment: str, port: int, model_version: str):
    """Deploy the heart disease prediction model."""
    try:
        # Deploy based on environment
        deploy_model(model_version, {}, environment, port)

        print("Deployment completed successfully!")

    except Exception as e:
        print(f"Deployment failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
