from flow import monitoring_flow
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule


def deploy_monitoring():
    """Deploy the monitoring flow."""
    # Create deployment that runs every hour
    deployment = Deployment.build_from_flow(
        flow=monitoring_flow,
        name="monitoring",
        schedule=CronSchedule(cron="0 * * * *"),  # Run every hour
        work_queue_name="default",
        tags=["monitoring"],
    )

    # Apply the deployment
    deployment.apply()


if __name__ == "__main__":
    deploy_monitoring()
