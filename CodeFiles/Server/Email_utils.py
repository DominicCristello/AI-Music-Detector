# ── email_utils.py ────────────────────────────────────────────────────────────
# Sends transactional email (password-reset codes). SERVER-SIDE ONLY.
#
# Development: use Gmail SMTP with an "App Password" (NOT your normal Google
# password). Create one at https://myaccount.google.com/apppasswords
# (requires 2-Step Verification enabled on the account).
#
# Production: switch to a transactional provider (SendGrid, Mailgun, SES, Resend)
# — Gmail rate-limits and isn't meant for volume. The send_email() interface
# below stays the same; only the transport changes.

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from Config import (SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD,
                    SMTP_FROM_NAME, EMAIL_ENABLED)


def send_email(to_email: str, subject: str, body: str,
               reply_to: str = None) -> tuple:
    """
    Send a plain-text email. Returns (success: bool, error: str | None).
    If EMAIL_ENABLED is False (no SMTP configured), prints to console instead
    so you can still test the flow without a real mailbox.
    """
    if not EMAIL_ENABLED:
        # Dev fallback: print the email so you can read the code in the server
        # terminal without configuring SMTP.
        print("\n" + "=" * 60)
        print(f"[EMAIL — dev mode, not actually sent]")
        print(f"To:      {to_email}")
        print(f"Subject: {subject}")
        print(f"Body:\n{body}")
        print("=" * 60 + "\n")
        return True, None

    try:
        msg = MIMEMultipart()
        msg['From'] = f"{SMTP_FROM_NAME} <{SMTP_USER}>"
        msg['To'] = to_email
        msg['Subject'] = subject
        if reply_to:
            msg['Reply-To'] = reply_to
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        return True, None
    except Exception as e:
        print(f"[email] send failed: {e}")
        return False, str(e)


def send_reset_code(to_email: str, code: str) -> tuple:
    """Send a password-reset code email."""
    subject = "Your AI Music Detector password reset code"
    body = (
        f"You requested a password reset.\n\n"
        f"Your six-digit code is: {code}\n\n"
        f"This code expires in 10 minutes. If you didn't request this, "
        f"you can safely ignore this email."
    )
    return send_email(to_email, subject, body)


def send_signup_code(to_email: str, code: str) -> tuple:
    """Send an email-verification code for a new signup."""
    subject = "Verify your AI Music Detector email"
    body = (
        f"Welcome to AI Music Detector!\n\n"
        f"Your six-digit verification code is: {code}\n\n"
        f"Enter it in the app to finish creating your account. This code "
        f"expires in 2 minutes. If you didn't request this, you can safely "
        f"ignore this email."
    )
    return send_email(to_email, subject, body)
