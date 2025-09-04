"""Prefect deployment configurations."""

from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule

from prefect_flows import monitoring_flow, training_flow

# Create training deployment
training_deployment = Deployment.build_from_flow(
    flow=training_flow,
    name="heart-disease-model-training",
    version=1,
    work_queue_name="default-queue",
    schedule=CronSchedule(cron="0 0 * * 0"),  # Run weekly on Sunday at midnight
    tags=["training", "production"],
)

# Create monitoring deployment
monitoring_deployment = Deployment.build_from_flow(
    flow=monitoring_flow,
    name="heart-disease-model-monitoring",
    version=1,
    work_queue_name="default-queue",
    schedule=CronSchedule(cron="0 * * * *"),  # Run hourly
    tags=["monitoring", "production"],
)

if __name__ == "__main__":
    # Apply deployments
    training_deployment.apply()
    monitoring_deployment.apply()
