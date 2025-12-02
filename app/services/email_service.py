import logging
import smtplib
from email.message import EmailMessage
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)


class EmailNotConfiguredError(RuntimeError):
    pass


def _require(value: Optional[str], name: str) -> str:
    if not value:
        raise EmailNotConfiguredError(f"Missing SMTP configuration value: {name}")
    return value


def send_email(
    *,
    to_email: str,
    subject: str,
    html_body: str,
    text_body: Optional[str] = "Password reset instructions",
) -> None:
    smtp_host = _require(getattr(settings, "smtp_host", None), "SMTP_HOST")
    smtp_port = getattr(settings, "smtp_port", None) or 587
    email_sender = _require(getattr(settings, "email_sender", None), "EMAIL_SENDER")
    smtp_username = getattr(settings, "smtp_username", None)
    smtp_password = getattr(settings, "smtp_password", None)
    use_tls = bool(getattr(settings, "smtp_use_tls", True))
    timeout = getattr(settings, "smtp_timeout", 30)

    if bool(smtp_username) ^ bool(smtp_password):
        raise EmailNotConfiguredError("SMTP username/password must both be provided")

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = email_sender
    message["To"] = to_email

    # Always include plain text + HTML for Gmail deliverability
    message.set_content(text_body or "")
    message.add_alternative(html_body, subtype="html")

    logger.debug("Connecting to SMTP server %s:%s (TLS=%s)", smtp_host, smtp_port, use_tls)

    try:
        with smtplib.SMTP(host=smtp_host, port=smtp_port, timeout=timeout) as server:
            server.set_debuglevel(0)
            if use_tls:
                server.starttls()
            if smtp_username and smtp_password:
                server.login(smtp_username, smtp_password)
            server.send_message(message)
    except smtplib.SMTPException as exc:
        logger.exception("Unable to send email via SMTP: %s", exc)
        raise

    logger.info("Password reset email sent to %s", to_email)
