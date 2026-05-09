"""
Infera AI — Gmail Sync Engine
===================================
Fetches from MULTIPLE Gmail folders so phishing in Spam is caught too.

Folders scanned:
  INBOX          — regular inbox (mostly safe, some phishing gets through)
  [Gmail]/Spam   — where Gmail puts suspected spam/phishing (most threats live here)

Pipeline per email:
  IMAP fetch → deduplicate by message_id → ML classify → save to DB → log alert
"""

from django.utils import timezone
from .imap_connector import GmailIMAPConnector
from .ml_engine import classify_email
from .models import Email, ScanLog, GmailAccount


# Folders to scan — order matters, Inbox first then Spam
FOLDERS_TO_SCAN = [
    ('INBOX',         'Inbox'),
    ('[Gmail]/Spam',  'Spam'),
]


def sync_gmail_account(account: GmailAccount, user=None) -> dict:
    """
    Full sync pipeline for one Gmail account.
    Scans INBOX + Spam folder so phishing emails are found.

    Returns a summary dict:
      {
        'success': True/False,
        'fetched': int,   total emails fetched across all folders
        'new': int,       newly scanned (not seen before)
        'skipped': int,   already in database
        'phishing': int,
        'suspicious': int,
        'safe': int,
        'errors': [],
        'folders': {      per-folder breakdown
          'Inbox': {'fetched': x, 'new': y, 'phishing': z},
          'Spam':  {'fetched': x, 'new': y, 'phishing': z},
        }
      }
    """
    summary = {
        'success':    False,
        'fetched':    0,
        'new':        0,
        'skipped':    0,
        'phishing':   0,
        'suspicious': 0,
        'safe':       0,
        'errors':     [],
        'folders':    {},
    }

    connector = GmailIMAPConnector(account.email_address, account.app_password)

    # ── Scan each folder ──────────────────────────────────────────────────
    for imap_folder, display_name in FOLDERS_TO_SCAN:
        folder_summary = {'fetched': 0, 'new': 0, 'phishing': 0,
                          'suspicious': 0, 'safe': 0}

        ok, result = connector.fetch_emails(
            limit=account.fetch_limit,
            folder=imap_folder
        )

        if not ok:
            # Spam folder might not exist on all accounts — that's OK
            summary['errors'].append(f'{display_name}: {result}')
            summary['folders'][display_name] = folder_summary
            continue

        raw_emails = result
        folder_summary['fetched'] = len(raw_emails)
        summary['fetched'] += len(raw_emails)

        # ── Process each email in this folder ─────────────────────────────
        for raw in raw_emails:
            try:
                message_id = raw.get('message_id', '').strip()

                # Deduplication — skip if already scanned
                if message_id and Email.objects.filter(message_id=message_id).exists():
                    summary['skipped'] += 1
                    continue

                sender      = raw.get('sender', '')
                subject     = raw.get('subject', '(No Subject)')
                body        = raw.get('body', '')
                urls        = raw.get('urls', [])
                attachments = raw.get('attachments', [])
                attachment_name = attachments[0] if attachments else ''

                # ── Run neural network classifier ──────────────────────────
                ml_result = classify_email(
                    sender=sender,
                    subject=subject,
                    body=body,
                    urls=urls,
                    attachment=attachment_name,
                )

                # ── Parse received date ────────────────────────────────────
                from email.utils import parsedate_to_datetime
                try:
                    received_at = parsedate_to_datetime(raw['date_str'])
                    if received_at.tzinfo is None:
                        from django.utils.timezone import make_aware
                        received_at = make_aware(received_at)
                except Exception:
                    received_at = timezone.now()

                # ── Save email to database ─────────────────────────────────
                # Note: we tag which folder it came from in why_flagged
                why = ml_result['why_flagged']
                if imap_folder != 'INBOX':
                    why = f'[Found in {display_name}] ' + why

                email_obj = Email.objects.create(
                    account          = account,
                    user             = user,
                    sender           = sender[:254],
                    sender_domain    = ml_result['sender_domain'],
                    subject          = subject[:499],
                    body             = body,
                    message_id       = message_id,
                    status           = ml_result['status'],
                    risk_score       = ml_result['risk_score'],
                    text_score       = ml_result['text_score'],
                    url_score        = ml_result['url_score'],
                    metadata_score   = ml_result['metadata_score'],
                    attachment_score = ml_result['attachment_score'],
                    extracted_urls   = urls,
                    has_attachment   = bool(attachments),
                    attachment_name  = attachment_name,
                    why_flagged      = why,
                    received_at      = received_at,
                )

                # ── Create alert log for threats ───────────────────────────
                status = ml_result['status']
                if status in ('phishing', 'suspicious'):
                    ScanLog.objects.create(
                        email   = email_obj,
                        level   = 'danger' if status == 'phishing' else 'warning',
                        message = (
                            f"[{status.upper()}] {subject[:60]} "
                            f"from {sender} — score {round(ml_result['risk_score']*100)}%"
                            f" (found in {display_name})"
                        ),
                    )

                # Update counts
                summary['new']    += 1
                summary[status]    = summary.get(status, 0) + 1
                folder_summary['new']      += 1
                folder_summary[status]      = folder_summary.get(status, 0) + 1

            except Exception as e:
                summary['errors'].append(
                    f'{display_name}: failed on one email — {str(e)}'
                )
                continue

        summary['folders'][display_name] = folder_summary

    # ── Update account metadata ───────────────────────────────────────────
    account.last_synced  = timezone.now()
    account.total_synced = Email.objects.filter(account=account).count()
    account.save()

    summary['success'] = True
    return summary
