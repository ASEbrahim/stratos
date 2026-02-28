"""
Email service for StratOS — SMTP-based email delivery.

Handles verification codes and password reset emails.
Gracefully degrades when SMTP is not configured.
"""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger("STRAT_OS")


class EmailService:
    """SMTP email service with graceful degradation."""

    def __init__(self, config: dict):
        self._config = config.get("email", {})

    def is_configured(self) -> bool:
        """Check if SMTP is properly configured."""
        return bool(self._config.get("smtp_host") and self._config.get("smtp_user"))

    def _send(self, to_email: str, subject: str, html_body: str):
        """Send an email via SMTP."""
        if not self.is_configured():
            raise RuntimeError("SMTP not configured")

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self._config.get("smtp_user")
        msg["To"] = to_email
        msg.attach(MIMEText(html_body, "html"))

        host = self._config["smtp_host"]
        port = self._config.get("smtp_port", 587)
        user = self._config["smtp_user"]
        password = self._config.get("smtp_password", "")

        with smtplib.SMTP(host, port, timeout=10) as server:
            server.starttls()
            server.login(user, password)
            server.send_message(msg)

        logger.info(f"Email sent to {to_email}: {subject}")

    def send_verification(self, email: str, code: str, display_name: str = ""):
        """Send a verification code email."""
        name = display_name or email.split("@")[0]
        subject = f"StratOS — Verification Code: {code}"
        body = f"""
        <div style="font-family: -apple-system, sans-serif; max-width: 480px; margin: 0 auto; padding: 32px;">
            <h2 style="color: #e2e8f0; margin-bottom: 8px;">Welcome to StratOS</h2>
            <p style="color: #94a3b8;">Hi {name}, your verification code is:</p>
            <div style="background: #1e293b; border: 1px solid #334155; border-radius: 12px;
                        padding: 24px; text-align: center; margin: 24px 0;">
                <span style="font-size: 32px; font-weight: 700; letter-spacing: 8px; color: #34d399;">
                    {code}
                </span>
            </div>
            <p style="color: #64748b; font-size: 13px;">
                This code expires in 15 minutes. If you didn't create a StratOS account, ignore this email.
            </p>
        </div>
        """
        self._send(email, subject, body)

    def send_reset_code(self, email: str, code: str, display_name: str = ""):
        """Send a password reset code email."""
        name = display_name or email.split("@")[0]
        subject = f"StratOS — Password Reset Code: {code}"
        body = f"""
        <div style="font-family: -apple-system, sans-serif; max-width: 480px; margin: 0 auto; padding: 32px;">
            <h2 style="color: #e2e8f0; margin-bottom: 8px;">Password Reset</h2>
            <p style="color: #94a3b8;">Hi {name}, your password reset code is:</p>
            <div style="background: #1e293b; border: 1px solid #334155; border-radius: 12px;
                        padding: 24px; text-align: center; margin: 24px 0;">
                <span style="font-size: 32px; font-weight: 700; letter-spacing: 8px; color: #fb7185;">
                    {code}
                </span>
            </div>
            <p style="color: #64748b; font-size: 13px;">
                This code expires in 15 minutes. If you didn't request a password reset, ignore this email.
            </p>
        </div>
        """
        self._send(email, subject, body)
