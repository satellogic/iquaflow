import smtplib


def smtpNotifier(to, subject, body):
    sender = to
    receivers = [to]
    message = (
        f"From:  <{to}>\n" f"To: {to} <{to}>\n" f"Subject: {subject}\n" f"{body}\n"
    )

    try:
        smtpObj = smtplib.SMTP("localhost")
        smtpObj.sendmail(sender, receivers, message)
        print(f"Successfully notified to {to}")
    except smtplib.SMTPException:
        print(f"Error: unable to notify to {to}")
