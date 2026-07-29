"""
Management command: python manage.py backfill_urls

Repairs emails that were synced before the HTML URL-extraction fix (f9bad81).

Why a re-sync does not do this
------------------------------
sync_gmail_account() deduplicates on message_id and skips anything already in
the database, so previously-synced rows are never revisited. And the fix cannot
be applied to what is already stored: the saved `body` is the *flattened* text,
produced after the tag-stripping regex had already deleted every href. The link
targets are simply not in the database, so the affected messages have to be
pulled from Gmail again.

Each repaired row is re-fetched by a server-side SEARCH on its Message-ID
(read-only, so nothing in the mailbox is marked read), re-parsed with the fixed
extractor, and re-classified. The search covers All Mail, Spam and Trash rather
than sync's Inbox+Spam: sync only ever wants newly-arrived mail, but a message
synced weeks ago has very likely been archived since, which in Gmail means it
is in All Mail and nowhere else.
Re-classification is not optional cosmetics: the old scores were produced from a
truncated body with no URLs, and by a feature vector that was misaligned with
the trained model, so leaving them in place would show phishing links next to a
verdict computed as though there were none. Pass --urls-only to fill in the
links and leave the existing verdicts untouched.

Usage:
    python manage.py backfill_urls --dry-run
    python manage.py backfill_urls
    python manage.py backfill_urls --account you@gmail.com
    python manage.py backfill_urls --all          # every row, not just link-less
"""
from django.core.management.base import BaseCommand
from django.db import transaction

import re

from detector.imap_connector import GmailIMAPConnector, normalize_message_id
from detector.ml_engine import classify_email
from detector.models import Email, ScanLog, GmailAccount

# The original sync recorded which folder it found a message in, as a
# '[Found in Spam] ' prefix on why_flagged. A backfill searches All Mail,
# which cannot tell us that, so the recorded value is preserved rather than
# recomputed — the sync's observation is the accurate one.
FOLDER_TAG_RE = re.compile(r'^\[Found in [^\]]+\]\s*')


class Command(BaseCommand):
    help = 'Re-fetch and repair emails synced before the HTML URL-extraction fix.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--account',
            help='Only repair this Gmail address (default: every active account).',
        )
        parser.add_argument(
            '--all',
            action='store_true',
            help=(
                'Re-process every stored email, not only the ones with no links. '
                'Also catches messages whose body came from a decoy text/plain '
                'part, which the default filter cannot detect.'
            ),
        )
        parser.add_argument(
            '--urls-only',
            action='store_true',
            help='Fill in links and body only; leave existing scores and verdicts alone.',
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Report what would change without writing to the database.',
        )

    def handle(self, *args, **options):
        accounts = GmailAccount.objects.filter(is_active=True)
        if options['account']:
            accounts = accounts.filter(email_address=options['account'])
            if not accounts.exists():
                self.stderr.write(
                    self.style.ERROR(
                        f"No active account matches {options['account']}"
                    )
                )
                return

        if not accounts.exists():
            self.stdout.write(self.style.WARNING('No active Gmail accounts to repair.'))
            return

        totals = {'repaired': 0, 'urls_found': 0, 'unmatched': 0, 'reclassified': 0}

        for account in accounts:
            self.stdout.write(self.style.HTTP_INFO(f'\n── {account.email_address} ──'))
            self._repair_account(account, options, totals)

        self.stdout.write('')
        verb = 'would repair' if options['dry_run'] else 'repaired'
        self.stdout.write(self.style.SUCCESS(
            f"Done: {verb} {totals['repaired']} email(s), "
            f"{totals['urls_found']} link(s) recovered, "
            f"{totals['reclassified']} verdict(s) changed."
        ))
        if totals['unmatched']:
            self.stdout.write(self.style.WARNING(
                f"{totals['unmatched']} email(s) could not be found in Gmail "
                f"(deleted, moved out of Inbox/Spam, or seeded rather than synced) "
                f"— these keep their current data."
            ))

    # ── Per-account work ─────────────────────────────────────────────────────

    def _repair_account(self, account, options, totals):
        candidates = self._candidates(account, options['all'])
        if not candidates:
            self.stdout.write('  Nothing to repair.')
            return

        # message_id is what we match on, and it is not unique-constrained, so
        # keep every row that shares one and repair them together.
        by_message_id = {}
        for email_obj in candidates:
            key = normalize_message_id(email_obj.message_id)
            if key:
                by_message_id.setdefault(key, []).append(email_obj)

        skipped_no_id = len(candidates) - sum(len(v) for v in by_message_id.values())
        if skipped_no_id:
            self.stdout.write(
                f'  {skipped_no_id} email(s) have no Message-ID and cannot be '
                f'matched back to Gmail — skipping.'
            )
            totals['unmatched'] += skipped_no_id

        self.stdout.write(f'  {len(by_message_id)} email(s) to look up in Gmail.')

        connector = GmailIMAPConnector(account.email_address, account.app_password)

        def progress(label, found, total):
            self.stdout.write(f'  searched {label}: {found}/{total} matched.')

        ok, result = connector.fetch_by_message_ids(by_message_id, progress=progress)
        if not ok:
            self.stderr.write(self.style.ERROR(f'  {result}'))
            return

        for key, (raw, label) in result.items():
            for email_obj in by_message_id[key]:
                self._apply(email_obj, raw, label, options, totals)

        missing = set(by_message_id) - set(result)
        if missing:
            totals['unmatched'] += sum(len(by_message_id[k]) for k in missing)

    def _candidates(self, account, include_all):
        """Rows worth re-fetching for this account."""
        rows = Email.objects.filter(account=account)
        if include_all:
            return list(rows)
        # The symptom the fix addresses: a stored email with no links at all.
        # Evaluated in Python rather than as a JSON lookup so it behaves the
        # same on SQLite and Postgres, and so legacy nulls are covered too.
        return [e for e in rows if not e.extracted_urls]

    def _apply(self, email_obj, raw, found_in, options, totals):
        """Write one re-fetched message back over its stored row."""
        urls = raw.get('urls', [])
        body = raw.get('body', '')
        attachments = raw.get('attachments', [])
        attachment_name = attachments[0] if attachments else ''

        old_status = email_obj.status
        new_status = old_status

        email_obj.body = body
        email_obj.extracted_urls = urls
        email_obj.has_attachment = bool(attachments)
        email_obj.attachment_name = attachment_name

        if not options['urls_only']:
            ml_result = classify_email(
                sender=raw.get('sender', email_obj.sender),
                subject=raw.get('subject', email_obj.subject),
                body=body,
                urls=urls,
                attachment=attachment_name,
            )
            new_status = ml_result['status']

            # Carry over the folder the original sync found it in. Finding it
            # in All Mail now says nothing about where it was delivered, and
            # dropping the tag would lose that the message came from Spam.
            existing_tag = FOLDER_TAG_RE.match(email_obj.why_flagged or '')
            why = ml_result['why_flagged']
            if existing_tag:
                why = existing_tag.group(0) + why

            email_obj.sender_domain    = ml_result['sender_domain']
            email_obj.status           = new_status
            email_obj.risk_score       = ml_result['risk_score']
            email_obj.text_score       = ml_result['text_score']
            email_obj.url_score        = ml_result['url_score']
            email_obj.metadata_score   = ml_result['metadata_score']
            email_obj.attachment_score = ml_result['attachment_score']
            email_obj.why_flagged      = why

        totals['repaired'] += 1
        totals['urls_found'] += len(urls)
        if new_status != old_status:
            totals['reclassified'] += 1

        change = f'{len(urls)} link(s) (via {found_in})'
        if new_status != old_status:
            change += f', {old_status} → {new_status}'
        self.stdout.write(f'    {"[dry-run] " if options["dry_run"] else ""}'
                          f'{email_obj.subject[:55]!r}: {change}')

        if options['dry_run']:
            return

        with transaction.atomic():
            email_obj.save()

            # A verdict that has just turned into a threat never produced an
            # alert during the original sync, so raise one now.
            if new_status != old_status and new_status in ('phishing', 'suspicious'):
                ScanLog.objects.create(
                    email   = email_obj,
                    level   = 'danger' if new_status == 'phishing' else 'warning',
                    message = (
                        f'[{new_status.upper()}] {email_obj.subject[:60]} '
                        f'from {email_obj.sender} — score '
                        f'{round(email_obj.risk_score * 100)}% '
                        f'(re-scanned after URL extraction fix)'
                    ),
                )
