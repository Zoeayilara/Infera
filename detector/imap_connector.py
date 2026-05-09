"""
Infera AI — IMAP Email Connector
======================================
This module connects to a user's Gmail account via IMAP (SSL on port 993).
It does NOT store the user's password anywhere — it only uses it for the
duration of the fetch session, then discards it.

How IMAP works:
  1. Open a secure SSL socket to imap.gmail.com:993
  2. Authenticate with email + app password
  3. SELECT the INBOX folder
  4. SEARCH for email IDs (we get the N most recent)
  5. FETCH each email's raw bytes
  6. Parse the raw bytes into subject, sender, body, attachments
  7. Close the connection
"""

import imaplib       # Python built-in: speaks IMAP protocol
import email         # Python built-in: parses raw email bytes into objects
import email.header  # for decoding encoded subjects like =?UTF-8?...
import re
from datetime import datetime


# ── Helpers ──────────────────────────────────────────────────────────────────

def decode_header_value(raw_value):
    """
    Email headers can be encoded like: =?UTF-8?B?VXJnZW50...?=
    This function decodes them back to plain readable text.
    """
    if not raw_value:
        return ''
    parts = email.header.decode_header(raw_value)
    decoded = []
    for part, charset in parts:
        if isinstance(part, bytes):
            try:
                decoded.append(part.decode(charset or 'utf-8', errors='replace'))
            except (LookupError, UnicodeDecodeError):
                decoded.append(part.decode('utf-8', errors='replace'))
        else:
            decoded.append(str(part))
    return ' '.join(decoded).strip()


def extract_body_and_attachments(msg):
    """
    An email can be 'multipart' (has both HTML + text + attachments)
    or 'simple' (just plain text).

    We walk through all parts and:
    - Collect the plain text body (preferred over HTML for analysis)
    - Collect attachment filenames
    """
    body_text = ''
    attachments = []

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = str(part.get('Content-Disposition') or '')

            # Skip multipart containers themselves
            if content_type == 'multipart/alternative':
                continue

            # It's an attachment if it has Content-Disposition: attachment
            if 'attachment' in disposition:
                filename = part.get_filename()
                if filename:
                    attachments.append(decode_header_value(filename))
                continue

            # Grab plain text body (better for ML than HTML)
            if content_type == 'text/plain' and not body_text:
                try:
                    charset = part.get_content_charset() or 'utf-8'
                    body_text = part.get_payload(decode=True).decode(
                        charset, errors='replace'
                    )
                except Exception:
                    body_text = str(part.get_payload())

            # Fall back to HTML if no plain text found
            elif content_type == 'text/html' and not body_text:
                try:
                    charset = part.get_content_charset() or 'utf-8'
                    html = part.get_payload(decode=True).decode(
                        charset, errors='replace'
                    )
                    # Strip HTML tags for cleaner text analysis
                    body_text = re.sub(r'<[^>]+>', ' ', html)
                    body_text = re.sub(r'\s+', ' ', body_text).strip()
                except Exception:
                    pass
    else:
        # Simple single-part email
        try:
            charset = msg.get_content_charset() or 'utf-8'
            body_text = msg.get_payload(decode=True).decode(
                charset, errors='replace'
            )
        except Exception:
            body_text = str(msg.get_payload())

    return body_text.strip(), attachments


def extract_urls_from_text(text):
    """Pull all URLs out of email body text."""
    return re.findall(r'https?://[^\s<>"\')\]]+', text)


# ── Main connector ────────────────────────────────────────────────────────────

class GmailIMAPConnector:
    """
    Connects to Gmail via IMAP SSL and fetches emails.

    Usage:
        connector = GmailIMAPConnector('you@gmail.com', 'abcd efgh ijkl mnop')
        ok, emails = connector.fetch_emails(limit=30)
    """

    IMAP_HOST = 'imap.gmail.com'
    IMAP_PORT = 993  # Always 993 for SSL

    def __init__(self, email_address: str, app_password: str):
        self.email_address = email_address.strip()
        # App passwords from Google come with spaces — remove them
        self.app_password = app_password.strip().replace(' ', '')

    def test_connection(self):
        """
        Just tries to login and immediately logout.
        Returns (True, 'OK') or (False, 'error message')
        Used on the settings page to verify credentials before saving.
        """
        try:
            # imaplib.IMAP4_SSL opens an encrypted connection automatically
            conn = imaplib.IMAP4_SSL(self.IMAP_HOST, self.IMAP_PORT)
            conn.login(self.email_address, self.app_password)
            conn.logout()
            return True, 'Connection successful'
        except imaplib.IMAP4.error as e:
            error = str(e)
            if 'AUTHENTICATIONFAILED' in error or 'Invalid credentials' in error:
                return False, 'Authentication failed. Check your email and App Password.'
            return False, f'IMAP error: {error}'
        except ConnectionRefusedError:
            return False, 'Could not reach Gmail servers. Check internet connection.'
        except Exception as e:
            return False, f'Unexpected error: {str(e)}'

    def fetch_emails(self, limit: int = 50, folder: str = 'INBOX'):
        """
        Connects to Gmail, fetches the most recent `limit` emails.

        Returns:
            (True, list_of_email_dicts) on success
            (False, error_message_string) on failure

        Each email dict contains:
            sender, subject, body, urls, attachments, date_str, message_id
        """
        try:
            # ── Step 1: Connect ───────────────────────────────────────────
            # This opens a TLS-encrypted socket to Google's IMAP server
            conn = imaplib.IMAP4_SSL(self.IMAP_HOST, self.IMAP_PORT)

            # ── Step 2: Authenticate ──────────────────────────────────────
            # Uses the app password, NOT the real Gmail password
            conn.login(self.email_address, self.app_password)

            # ── Step 3: Select folder ─────────────────────────────────────
            # 'INBOX' is the main inbox. You could also use '[Gmail]/Spam' etc.
            status, messages = conn.select(folder)
            if status != 'OK':
                return False, f'Could not open {folder}'

            # ── Step 4: Search for all email IDs ──────────────────────────
            # 'ALL' returns every email ID in the folder as a list of numbers
            status, data = conn.search(None, 'ALL')
            if status != 'OK':
                return False, 'Could not search inbox'

            # data[0] is a space-separated byte string of IDs like b'1 2 3 4 5'
            all_ids = data[0].split()

            # Take only the most recent N emails (IDs are in ascending order)
            recent_ids = all_ids[-limit:] if len(all_ids) > limit else all_ids
            # Reverse so newest comes first
            recent_ids = list(reversed(recent_ids))

            # ── Step 5: Fetch and parse each email ────────────────────────
            parsed_emails = []

            for email_id in recent_ids:
                try:
                    # RFC822 means "give me the full raw email bytes"
                    status, msg_data = conn.fetch(email_id, '(RFC822)')
                    if status != 'OK':
                        continue

                    # msg_data[0][1] is the raw email bytes
                    raw_bytes = msg_data[0][1]

                    # email.message_from_bytes parses raw bytes into an
                    # email.Message object with .get(), .walk() etc.
                    msg = email.message_from_bytes(raw_bytes)

                    # Decode the headers (they can be encoded in various charsets)
                    sender = decode_header_value(msg.get('From', ''))
                    subject = decode_header_value(msg.get('Subject', '(No Subject)'))
                    date_str = msg.get('Date', '')
                    message_id = msg.get('Message-ID', str(email_id))

                    # Extract plain text body and attachment filenames
                    body, attachments = extract_body_and_attachments(msg)

                    # Extract URLs from body
                    urls = extract_urls_from_text(body)

                    # Clean up sender — extract just the email address
                    # e.g. "John Smith <john@example.com>" → "john@example.com"
                    sender_email_match = re.search(r'<([^>]+)>', sender)
                    sender_clean = sender_email_match.group(1) if sender_email_match else sender

                    parsed_emails.append({
                        'sender': sender_clean,
                        'sender_display': sender,
                        'subject': subject or '(No Subject)',
                        'body': body[:5000],  # cap at 5000 chars for ML
                        'urls': urls,
                        'attachments': attachments,
                        'date_str': date_str,
                        'message_id': message_id,
                    })

                except Exception:
                    # If one email fails to parse, skip it and continue
                    continue

            # ── Step 6: Close connection ──────────────────────────────────
            conn.close()
            conn.logout()

            return True, parsed_emails

        except imaplib.IMAP4.error as e:
            error = str(e)
            if 'AUTHENTICATIONFAILED' in error or 'Invalid credentials' in error:
                return False, 'Authentication failed. Check your Gmail App Password.'
            return False, f'Gmail connection error: {error}'
        except Exception as e:
            return False, f'Unexpected error: {str(e)}'
