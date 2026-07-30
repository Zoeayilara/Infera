"""
Management command: python manage.py rescore_emails

Re-scores stored emails through the current engine, without touching Gmail.

Why this is not backfill_urls
-----------------------------
backfill_urls repairs rows whose data was *destroyed* before storage: the
HTML-stripping regex deleted every href before the body was saved, so the link
targets were not in the database and the messages had to be pulled from Gmail
again. That is not this problem. Here every input classify_email() needs is
already stored — sender, subject, body, extracted_urls, attachment_name — and
only the engine's derivation has changed. So this needs no network, no
Message-ID, and no GmailAccount, and it reaches rows backfill_urls cannot:
backfill_urls filters on account=<account>, which excludes every seeded row.

What it is for
--------------
A stored verdict was produced by whatever the engine looked like when the row
was written. When the engine changes, stored rows keep their old verdicts while
new syncs use the new logic, and the table holds two regimes at once. This
command brings the old rows forward.

Direction is not symmetric. Raising a verdict is safe: it flags mail that
should have been flagged. Lowering one silently un-flags mail someone may
already have acted on, so a default run never does it — de-escalations are
listed and counted, and applying them takes --apply-de-escalations. The tool
can assert "at least this bad" without being able to quietly say "actually,
fine".

Usage:
    python manage.py rescore_emails --dry-run
    python manage.py rescore_emails
    python manage.py rescore_emails --account you@gmail.com
    python manage.py rescore_emails --apply-de-escalations
"""
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

import re

from detector.ml_engine import classify_email, _load_model
from detector.models import Email, ScanLog

from ._console import make_console_tolerant

# Same tag backfill_urls preserves: the original sync recorded which folder the
# message arrived in as a why_flagged prefix. Re-scoring cannot rediscover it,
# so it is carried over rather than dropped.
FOLDER_TAG_RE = re.compile(r'^\[Found in [^\]]+\]\s*')

SEVERITY = {'safe': 0, 'pending': 0, 'suspicious': 1, 'phishing': 2}


class Command(BaseCommand):
    help = 'Re-score stored emails through the current engine (no Gmail access).'

    def add_arguments(self, parser):
        parser.add_argument(
            '--account',
            help='Only re-score this Gmail address (default: every email, '
                 'including seeded rows with no account).',
        )
        parser.add_argument(
            '--apply-de-escalations',
            action='store_true',
            help='Also apply verdicts that become less severe. Off by default: '
                 'lowering a verdict un-flags mail that may already have been '
                 'acted on.',
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Report what would change without writing to the database.',
        )

    def handle(self, *args, **options):
        # Email subjects are interpolated into the progress lines below and are
        # not guaranteed ASCII, so let them degrade rather than kill the run
        # when stdout is redirected to a non-UTF-8 stream. See _console.py.
        make_console_tolerant(self.stdout, self.stderr)

        # Refuse rather than silently re-scoring the whole table through the
        # rule-based fallback. That would swap model verdicts for rule verdicts
        # on every row, and afterwards nothing in the data would record that it
        # happened — a far bigger regime change than the one being repaired.
        # Deliberately not overridable.
        if not _load_model():
            raise CommandError(
                'Refusing to run: the trained model could not be loaded, so '
                'every row would be re-scored by the rule-based fallback and '
                'nothing in the data would record that. Train the model first '
                '(python ml/train_model.py), then re-run.'
            )

        rows = Email.objects.all()
        if options['account']:
            rows = rows.filter(account__email_address=options['account'])
            if not rows.exists():
                raise CommandError(
                    f"No emails found for account {options['account']}"
                )
        rows = list(rows.order_by('id'))

        if not rows:
            self.stdout.write(self.style.WARNING('No emails to re-score.'))
            return

        self.stdout.write(f'Re-scoring {len(rows)} email(s).')

        escalations, de_escalations, unchanged = [], [], 0

        for email_obj in rows:
            result = classify_email(
                sender     = email_obj.sender,
                subject    = email_obj.subject,
                body       = email_obj.body,
                urls       = email_obj.extracted_urls or [],
                attachment = email_obj.attachment_name or '',
            )
            old = email_obj.status
            new = result['status']

            if new == old:
                unchanged += 1
                # Scores can drift without the label moving; keep them current
                # so the page's numbers match the verdict they sit beside.
                self._write(email_obj, result, options, log_as=None)
                continue

            if SEVERITY[new] > SEVERITY[old]:
                escalations.append((email_obj, result, old, new))
            else:
                de_escalations.append((email_obj, result, old, new))

        # ── Escalations: always applied ──────────────────────────────────────
        if escalations:
            self.stdout.write('')
            self.stdout.write(self.style.HTTP_INFO('Escalations (applied):'))
            for email_obj, result, old, new in escalations:
                self._report(email_obj, result, old, new, options)
                self._write(email_obj, result, options, log_as=new)

        # ── De-escalations: listed, applied only on request ──────────────────
        if de_escalations:
            self.stdout.write('')
            applied = options['apply_de_escalations']
            header = ('De-escalations (applied - verdicts lowered):' if applied
                      else 'De-escalations (NOT applied):')
            self.stdout.write(self.style.WARNING(header))
            for email_obj, result, old, new in de_escalations:
                self._report(email_obj, result, old, new, options)
                if applied:
                    self._write(email_obj, result, options, log_as=new)

        # ── Summary ──────────────────────────────────────────────────────────
        verb = 'would change' if options['dry_run'] else 'changed'
        self.stdout.write('')
        self.stdout.write(self.style.SUCCESS(
            f'Done: {len(escalations)} escalation(s) {verb}, '
            f'{unchanged} unchanged.'
        ))
        if de_escalations and not options['apply_de_escalations']:
            self.stdout.write(self.style.WARNING(
                f'{len(de_escalations)} row(s) now score lower and were left '
                f'as they are. These are mail that is currently flagged and '
                f'would become less severe - review them, then re-run with '
                f'--apply-de-escalations to apply.'
            ))
        elif de_escalations:
            self.stdout.write(self.style.WARNING(
                f'{len(de_escalations)} verdict(s) lowered.'
            ))

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _report(self, email_obj, result, old, new, options):
        prefix = '[dry-run] ' if options['dry_run'] else ''
        # ASCII arrows: this runs on Windows consoles under cp1252, which cannot
        # encode U+2192 and raises rather than degrading.
        self.stdout.write(
            f'  {prefix}id={email_obj.id} {email_obj.subject[:45]!r}: '
            f'{old} -> {new} '
            f'({round(email_obj.risk_score * 100)}% -> '
            f'{round(result["risk_score"] * 100)}%)'
        )

    def _write(self, email_obj, result, options, log_as):
        """Write one re-scored row back. log_as is the new status, or None."""
        if options['dry_run']:
            return

        # Carry over the folder tag the original sync recorded; re-scoring has
        # no way to rediscover which mailbox the message arrived in.
        existing_tag = FOLDER_TAG_RE.match(email_obj.why_flagged or '')
        why = result['why_flagged']
        if existing_tag:
            why = existing_tag.group(0) + why

        email_obj.sender_domain    = result['sender_domain']
        email_obj.status           = result['status']
        email_obj.risk_score       = result['risk_score']
        email_obj.text_score       = result['text_score']
        email_obj.url_score        = result['url_score']
        email_obj.metadata_score   = result['metadata_score']
        email_obj.attachment_score = result['attachment_score']
        email_obj.why_flagged      = why

        with transaction.atomic():
            email_obj.save()

            if log_as in ('phishing', 'suspicious'):
                ScanLog.objects.create(
                    email   = email_obj,
                    level   = 'danger' if log_as == 'phishing' else 'warning',
                    message = (
                        f'[{log_as.upper()}] {email_obj.subject[:60]} '
                        f'from {email_obj.sender} — score '
                        f'{round(email_obj.risk_score * 100)}% '
                        f'(re-scored by rescore_emails)'
                    ),
                )
            elif log_as == 'safe':
                ScanLog.objects.create(
                    email   = email_obj,
                    level   = 'info',
                    message = (
                        f'[SAFE] {email_obj.subject[:60]} '
                        f'from {email_obj.sender} — verdict lowered to safe at '
                        f'{round(email_obj.risk_score * 100)}% '
                        f'(re-scored by rescore_emails --apply-de-escalations)'
                    ),
                )
