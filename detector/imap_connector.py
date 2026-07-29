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


def normalize_message_id(raw_value):
    """
    Canonical form of a Message-ID for comparison.

    Headers can arrive folded across lines and with stray padding, so the same
    ID can be stored one way and fetched back another. Collapsing whitespace
    makes both sides comparable.
    """
    if not raw_value:
        return ''
    return re.sub(r'\s+', '', str(raw_value)).strip()


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

    @staticmethod
    def parse_message(raw_bytes, fallback_id=''):
        """
        Turn one message's raw RFC822 bytes into the dict the rest of the
        pipeline consumes. Shared by fetch_emails() and fetch_by_message_ids()
        so a re-fetch produces exactly what a fresh sync would have produced.
        """
        msg = email.message_from_bytes(raw_bytes)

        # Decode the headers (they can be encoded in various charsets)
        sender = decode_header_value(msg.get('From', ''))
        subject = decode_header_value(msg.get('Subject', '(No Subject)'))
        date_str = msg.get('Date', '')
        message_id = msg.get('Message-ID', str(fallback_id))

        # Extract body text, attachment filenames, and every URL in the
        # message. URLs come back from here rather than being re-derived from
        # `body`, because flattening HTML discards the tags the link targets
        # live in.
        body, attachments, urls = extract_body_and_attachments(msg)

        # Clean up sender — extract just the email address
        # e.g. "John Smith <john@example.com>" → "john@example.com"
        sender_email_match = re.search(r'<([^>]+)>', sender)
        sender_clean = sender_email_match.group(1) if sender_email_match else sender

        return {
            'sender': sender_clean,
            'sender_display': sender,
            'subject': subject or '(No Subject)',
            'body': body[:5000],  # cap at 5000 chars for ML
            'urls': urls,
            'attachments': attachments,
            'date_str': date_str,
            'message_id': message_id,
        }

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

                    parsed_emails.append(
                        self.parse_message(raw_bytes, fallback_id=email_id)
                    )

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

    # ── Locating messages that are no longer in the Inbox ────────────────────

    # Folders to search when repairing already-synced rows, in the order they
    # are tried. Each entry is (special-use attribute, English fallback name,
    # label). The attribute is what we actually match on: Gmail localises the
    # display names ('[Gmail]/All Mail' is '[Gmail]/Tous les messages' on a
    # French account), but advertises stable SPECIAL-USE flags in its LIST
    # response, so the flag works on any locale.
    #
    # \All is Gmail's All Mail, which holds every message that is not spam or
    # trash — including everything the user has archived out of the Inbox.
    BACKFILL_FOLDERS = [
        (rb'\All',   '[Gmail]/All Mail', 'All Mail'),
        (rb'\Junk',  '[Gmail]/Spam',     'Spam'),
        (rb'\Trash', '[Gmail]/Trash',    'Trash'),
    ]

    # A stalled read would otherwise block the command forever: imaplib leaves
    # the socket timeout at None, and a backfill is typically run unattended.
    TIMEOUT_SECONDS = 60

    LIST_LINE_RE = re.compile(rb'^\((?P<attrs>[^)]*)\)\s+"[^"]*"\s+(?P<name>.+)$')

    def _resolve_folders(self, conn):
        """
        Work out this account's real folder names from its LIST response.

        Returns a list of (imap_folder_name, label). Falls back to the English
        Gmail names for any special use the server does not advertise, and
        always includes INBOX last as a backstop for non-Gmail servers that
        expose no \\All folder at all.
        """
        by_attribute = {}
        status, data = conn.list()
        if status == 'OK':
            for line in data or []:
                if isinstance(line, tuple):
                    line = b' '.join(part for part in line if part)
                match = self.LIST_LINE_RE.match((line or b'').strip())
                if not match:
                    continue
                name = match.group('name').strip().strip(b'"')
                for attribute in match.group('attrs').split():
                    by_attribute[attribute] = name.decode('utf-8', errors='replace')

        folders = []
        for attribute, fallback, label in self.BACKFILL_FOLDERS:
            folders.append((by_attribute.get(attribute, fallback), label))

        if not any(attribute in by_attribute for attribute, _, _ in self.BACKFILL_FOLDERS):
            folders.append(('INBOX', 'Inbox'))
        return folders

    @staticmethod
    def _quote(value):
        """Quote a string for use as an IMAP SEARCH argument."""
        escaped = value.replace('\\', '\\\\').replace('"', '\\"')
        return f'"{escaped}"'

    def fetch_by_message_ids(self, message_ids, progress=None):
        """
        Re-fetch specific already-known messages, looked up by Message-ID.

        fetch_emails() only ever returns the newest N messages of a folder,
        which is no use for repairing rows synced some time ago — and those
        rows may not be in the Inbox at all any more, since archiving a
        message moves it to All Mail.

        Lookup is done with a server-side SEARCH per Message-ID rather than by
        pulling every header in the folder. That matters: the header-scan
        approach costs the same whether twelve rows need repair or twelve
        thousand, and on All Mail — the folder that actually has to be
        searched — it means transferring a header for every message the
        account has ever received. SEARCH is O(rows being repaired) instead,
        and each response is a handful of sequence numbers.

        Folders are opened read-only, so a repair never marks a message as
        read or otherwise disturbs the mailbox.

        `progress` is an optional callable taking (label, found, total),
        called once per folder.

        Returns:
            (True, {normalized_message_id: (parsed_email_dict, folder_label)})
            (False, error_message_string) on failure
        """
        outstanding = {normalize_message_id(m) for m in message_ids}
        outstanding.discard('')
        if not outstanding:
            return True, {}

        total = len(outstanding)
        found = {}
        conn = None
        selected = False

        try:
            conn = imaplib.IMAP4_SSL(self.IMAP_HOST, self.IMAP_PORT,
                                     timeout=self.TIMEOUT_SECONDS)
            conn.login(self.email_address, self.app_password)

            for folder, label in self._resolve_folders(conn):
                if not outstanding:
                    break

                if selected:
                    conn.close()
                    selected = False

                status, _ = conn.select(folder, readonly=True)
                if status != 'OK':
                    # A folder the account does not have is not an error —
                    # Trash and Spam are both absent on some configurations.
                    continue
                selected = True

                for message_id in list(outstanding):
                    parsed = self._fetch_one(conn, message_id)
                    if parsed is not None:
                        found[message_id] = (parsed, label)
                        outstanding.discard(message_id)

                if progress:
                    progress(label, len(found), total)

            return True, found

        except imaplib.IMAP4.error as e:
            error = str(e)
            if 'AUTHENTICATIONFAILED' in error or 'Invalid credentials' in error:
                return False, 'Authentication failed. Check your Gmail App Password.'
            return False, f'Gmail connection error: {error}'
        except OSError as e:
            # Covers socket.timeout, which is an OSError subclass
            return False, f'Gmail connection failed or timed out: {e}'
        except Exception as e:
            return False, f'Unexpected error: {str(e)}'
        finally:
            if conn is not None:
                try:
                    if selected:
                        conn.close()
                    conn.logout()
                except Exception:
                    pass

    def _fetch_one(self, conn, message_id):
        """
        Find one message in the currently-selected folder by Message-ID and
        return its parsed form, or None if it is not in this folder.
        """
        try:
            status, data = conn.search(
                None, 'HEADER', 'Message-ID', self._quote(message_id)
            )
            if status != 'OK' or not data or not data[0]:
                return None

            # A Message-ID should be unique, but duplicates do occur (the same
            # message delivered twice); the first hit is as good as any.
            seq = data[0].split()[0]

            status, msg_data = conn.fetch(seq, '(BODY.PEEK[])')
            if status != 'OK':
                return None

            for item in msg_data:
                if not isinstance(item, tuple) or len(item) < 2:
                    continue
                parsed = self.parse_message(item[1])
                # Guard against a server returning a near-match rather than
                # the exact header we asked for.
                if normalize_message_id(parsed['message_id']) == message_id:
                    return parsed
            return None
        except Exception:
            # One unfindable message must not abort the whole backfill
            return None
