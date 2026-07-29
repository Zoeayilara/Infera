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
import html as html_lib  # for unescaping &amp; &nbsp; etc. out of HTML bodies
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


# A bare URL sitting in visible text, e.g. "verify at http://evil.example/go"
URL_TEXT_RE = re.compile(r'https?://[^\s<>"\')\]]+')

# A link target sitting in a tag attribute, e.g. <a href="http://evil.example">.
# Handles double-quoted, single-quoted and unquoted attribute values.
HTML_LINK_ATTR_RE = re.compile(
    r'(?:href|src)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^\s>]+))',
    re.IGNORECASE,
)

# <script>/<style> bodies are markup plumbing, not something the user ever
# reads — their contents would otherwise pollute the text sent to the model.
SCRIPT_STYLE_RE = re.compile(
    r'<(script|style)\b[^>]*>.*?</\1>', re.IGNORECASE | re.DOTALL
)


def _decode_part(part):
    """Decode one MIME part's payload into a string, whatever its charset."""
    try:
        charset = part.get_content_charset() or 'utf-8'
        return part.get_payload(decode=True).decode(charset, errors='replace')
    except Exception:
        return str(part.get_payload())


def _html_to_text(html_source):
    """
    Flatten HTML into readable text for the ML text model.

    IMPORTANT: this throws away every tag, and a link's target lives *inside*
    its tag. Always harvest URLs with extract_urls_from_html() on the raw
    markup BEFORE calling this — afterwards they are gone.
    """
    text = SCRIPT_STYLE_RE.sub(' ', html_source)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = html_lib.unescape(text)          # &amp; &nbsp; &#39; → & <space> '
    return re.sub(r'\s+', ' ', text).strip()


def _dedupe(urls):
    """Drop duplicates while keeping the order they appeared in the email."""
    seen = set()
    unique = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique.append(url)
    return unique


def extract_body_and_attachments(msg):
    """
    An email can be 'multipart' (has both HTML + text + attachments)
    or 'simple' (just plain text).

    We walk through all parts and:
    - Collect the plain text body (preferred over HTML for analysis)
    - Collect the HTML source, so links can be harvested from it
    - Collect attachment filenames

    Returns (body_text, attachments, urls).
    """
    plain_source = ''
    html_source = ''
    attachments = []

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = str(part.get('Content-Disposition') or '')

            # Skip multipart containers themselves — mixed/related/alternative
            if content_type.startswith('multipart/'):
                continue

            # It's an attachment if it has Content-Disposition: attachment
            if 'attachment' in disposition:
                filename = part.get_filename()
                if filename:
                    attachments.append(decode_header_value(filename))
                continue

            # Keep the two representations separately. We must not let a
            # text/plain part shadow the HTML one: senders routinely ship a
            # stub plain part ("Please enable HTML to view this message")
            # while every link lives only in the HTML alternative.
            if content_type == 'text/plain' and not plain_source:
                plain_source = _decode_part(part)
            elif content_type == 'text/html' and not html_source:
                html_source = _decode_part(part)
    else:
        # Simple single-part email
        payload = _decode_part(msg)
        if msg.get_content_type() == 'text/html':
            html_source = payload
        else:
            plain_source = payload

    # Harvest links from the raw HTML first, while the tags still exist.
    urls = extract_urls_from_text(plain_source)
    urls += extract_urls_from_html(html_source)

    # Body for the ML text model: the flattened HTML whenever there is one,
    # falling back to the plain part. The HTML alternative is what the
    # recipient's mail client actually renders, so it is the text the victim
    # reads — and unlike the plain part it cannot be sandbagged, since an
    # attacker is free to stuff a harmless decoy into a text/plain
    # alternative that no one will ever see.
    html_text = _html_to_text(html_source) if html_source else ''
    body_text = html_text or plain_source

    return body_text.strip(), attachments, _dedupe(urls)


def extract_urls_from_text(text):
    """Pull all URLs out of email body text."""
    if not text:
        return []
    return URL_TEXT_RE.findall(text)


def extract_urls_from_html(html_source):
    """
    Pull all URLs out of raw HTML, before any tag stripping happens.

    Two places a link can hide:
      1. href="..." / src="..." attributes — the usual case for phishing,
         where the visible text says "Click here" and the real target is
         only ever present inside the tag.
      2. A bare URL typed into the visible body text.

    Attribute values are unescaped, since query strings arrive as
    ...?id=1&amp;token=abc and would otherwise be captured with the entity.
    """
    if not html_source:
        return []

    urls = []
    for match in HTML_LINK_ATTR_RE.finditer(html_source):
        raw_value = match.group(1) or match.group(2) or match.group(3) or ''
        value = html_lib.unescape(raw_value).strip()
        # mailto:, tel:, cid: (inline images) and data: URIs are not web links
        if value.lower().startswith(('http://', 'https://')):
            urls.append(value)

    # Bare URLs in the visible text, after entities are resolved
    urls += extract_urls_from_text(_html_to_text(html_source))

    return _dedupe(urls)


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

                    # Extract body text, attachment filenames, and every URL
                    # in the message. URLs come back from here rather than
                    # being re-derived from `body`, because flattening HTML
                    # discards the tags the link targets live in.
                    body, attachments, urls = extract_body_and_attachments(msg)

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
