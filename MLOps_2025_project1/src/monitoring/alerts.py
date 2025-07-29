import json
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional

import requests
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


class AlertManager:
    """Manages alerts for model monitoring."""

    def __init__(
        self,
        slack_token: Optional[str] = None,
        email_config: Optional[Dict[str, str]] = None,
    ):
        """Initialize AlertManager with optional Slack and email configurations."""
        self.slack_token = slack_token or os.getenv("SLACK_TOKEN")
        self.email_config = email_config or {
            "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
            "smtp_port": int(os.getenv("SMTP_PORT", "587")),
            "sender_email": os.getenv("SENDER_EMAIL", ""),
            "sender_password": os.getenv("SENDER_PASSWORD", ""),
        }

        # Initialize Slack client if token is provided
        self.slack_client = (
            WebClient(token=self.slack_token) if self.slack_token else None
        )

    def format_alert_message(
        self, violations: List[str], metrics: Dict[str, float]
    ) -> str:
        """Format alert message with violations and metrics."""
        message = "⚠️ Model Monitoring Alert ⚠️\n\n"
        message += "The following violations were detected:\n"
        for violation in violations:
            message += f"• {violation}\n"

        message += "\nCurrent Metrics:\n"
        for metric, value in metrics.items():
            message += f"• {metric}: {value:.3f}\n"

        return message

    def send_slack_alert(self, message: str, channel: str = "#monitoring") -> bool:
        """
        Send an alert to Slack.

        Args:
            message: The message to send
            channel: The Slack channel to send to

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.slack_client:
            print("Slack client not configured")
            return False

        try:
            self.slack_client.chat_postMessage(channel=channel, text=message)
            return True
        except SlackApiError as e:
            print(f"Error sending Slack alert: {str(e)}")
            return False

    def send_email_alert(self, subject: str, body: str, recipient_email: str) -> bool:
        """
        Send an alert via email.

        Args:
            subject: Email subject
            body: Email body
            recipient_email: Recipient's email address

        Returns:
            bool: True if successful, False otherwise
        """
        if not all(self.email_config.values()):
            print("Email configuration incomplete")
            return False

        try:
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = self.email_config["sender_email"]
            msg["To"] = recipient_email

            with smtplib.SMTP(
                self.email_config["smtp_server"], self.email_config["smtp_port"]
            ) as server:
                server.starttls()
                server.login(
                    self.email_config["sender_email"],
                    self.email_config["sender_password"],
                )
                server.send_message(msg)
            return True
        except Exception as e:
            print(f"Error sending email alert: {str(e)}")
            return False

    def send_alert(
        self,
        message: str,
        alert_type: str = "slack",
        email_recipient: Optional[str] = None,
        slack_channel: str = "#monitoring",
    ) -> bool:
        """
        Send an alert through the specified channel.

        Args:
            message: The alert message
            alert_type: Type of alert ('slack' or 'email')
            email_recipient: Recipient email for email alerts
            slack_channel: Slack channel for Slack alerts

        Returns:
            bool: True if successful, False otherwise
        """
        if alert_type == "slack":
            return self.send_slack_alert(message, slack_channel)
        elif alert_type == "email" and email_recipient:
            return self.send_email_alert(
                "Model Monitoring Alert", message, email_recipient
            )
        else:
            print(f"Invalid alert type: {alert_type}")
            return False

    def trigger_alert(self, violations: List[str], metrics: Dict[str, float]) -> None:
        """Trigger alerts through all configured channels."""
        message = self.format_alert_message(violations, metrics)

        # Send email alert
        email_sent = self.send_email_alert(
            subject="Model Monitoring Alert", message=message
        )

        # Send Slack alert
        slack_sent = self.send_slack_alert(message)

        if not (email_sent or slack_sent):
            print("Failed to send alerts through any channel")
