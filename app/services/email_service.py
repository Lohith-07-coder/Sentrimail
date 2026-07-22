"""
SentriMail Email Service
------------------------
SMTP notification service for sending complaint resolution emails.
"""

import os
import smtplib
from email.mime.text import MIMEText
from deep_translator import GoogleTranslator


def send_resolution_email(complaint_data: dict, reply_text: str) -> None:
    email = complaint_data.get("email")
    if not email:
        return

    code = complaint_data.get("complaint_code", "N/A")
    orig_lang = complaint_data.get("original_language", "en")
    desc = complaint_data.get("description", "N/A")

    subject = f"Your complaint {code} has been resolved — SentriMail"
    body = f"Complaint Summary:\n{desc}\n\nOur Reply:\n{reply_text}"

    if orig_lang and orig_lang != "en":
        try:
            subject = GoogleTranslator(source='en', target=orig_lang).translate(subject)
            body = GoogleTranslator(source='en', target=orig_lang).translate(body)
        except Exception:
            pass

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = os.environ.get("MAIL_DEFAULT_SENDER", "noreply@sentrimail.com")
    msg["To"] = email

    try:
        server = smtplib.SMTP(
            os.environ.get("MAIL_SERVER", "localhost"),
            int(os.environ.get("MAIL_PORT", 587))
        )
        if os.environ.get("MAIL_USE_TLS") == "True":
            server.starttls()
            server.login(
                os.environ.get("MAIL_USERNAME", ""),
                os.environ.get("MAIL_PASSWORD", "")
            )
        server.send_message(msg)
        server.quit()
    except Exception as e:
        print(f"Email failed to send: {e}")
