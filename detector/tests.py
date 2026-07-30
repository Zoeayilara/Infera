"""
Infera AI — Email parsing tests
===============================
Regression tests for the IMAP body/URL extraction path.

The bug these guard against: HTML emails were flattened with a tag-stripping
regex *before* URLs were pulled out of the text. A link target lives inside
the tag (<a href="http://evil.example">Click here</a>), so stripping tags
deleted every URL and the classifier scored HTML phishing with no URL signal
at all. Plain-text mail was unaffected, which is what made it hard to spot.

Run: python manage.py test detector
"""

import email
import inspect
import io
import re
from io import StringIO
from unittest import mock

from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import SimpleTestCase, TestCase

from ml.train_model import extract_hand_crafted_features as training_features

from .imap_connector import (
    GmailIMAPConnector,
    extract_body_and_attachments,
    extract_urls_from_html,
    extract_urls_from_text,
    normalize_message_id,
)
from .management.commands.rescore_emails import SEVERITY
from .models import Email, GmailAccount, ScanLog
from .ml_engine import (
    RISK_PHISHING,
    RISK_SUSPICIOUS,
    _escalate,
    _feature_text,
    classify_email,
    extract_hand_crafted_features as serving_features,
    status_from_risk,
)


def parse(raw: bytes):
    """Run raw email bytes through the same path fetch_emails() uses."""
    return extract_body_and_attachments(email.message_from_bytes(raw))


# ── Message fixtures ─────────────────────────────────────────────────────────

SINGLE_PART_PLAIN = b"""From: PayPal Security <alerts@paypa1-secure.xyz>
Subject: Your account has been suspended
Content-Type: text/plain; charset="utf-8"

Dear Customer,
Unusual activity was detected. Verify now at
http://paypa1-verify.xyz/login?id=88
"""

SINGLE_PART_HTML = b"""From: PayPal Security <alerts@paypa1-secure.xyz>
Subject: Your account has been suspended
MIME-Version: 1.0
Content-Type: text/html; charset="utf-8"

<html><body>
<p>Dear Customer,</p>
<p><a href="http://paypa1-verify.xyz/login?id=88">Click here to verify now</a></p>
</body></html>
"""

MULTIPART_HTML_ONLY = b"""From: PayPal <s@paypa1-secure.xyz>
Subject: Account suspended
MIME-Version: 1.0
Content-Type: multipart/alternative; boundary="B"

--B
Content-Type: text/html; charset="utf-8"

<html><body><p>Dear Customer, your account is suspended.</p>
<a href="http://paypa1-verify.xyz/login">Verify now</a></body></html>
--B--
"""

MULTIPART_PLAIN_STUB = b"""From: DHL <no-reply@dhl-delivery.top>
Subject: Package on hold
MIME-Version: 1.0
Content-Type: multipart/alternative; boundary="B"

--B
Content-Type: text/plain; charset="utf-8"

Please enable HTML to view this message.
--B
Content-Type: text/html; charset="utf-8"

<html><body><p>Dear User, your package is on hold. Confirm now to release it:
<a href="http://dhl-redelivery.click/pay">Update now</a></p></body></html>
--B--
"""

MULTIPART_MIXED_WITH_ATTACHMENT = b"""From: DHL <a@dhl-delivery.top>
Subject: Invoice
MIME-Version: 1.0
Content-Type: multipart/mixed; boundary="B"

--B
Content-Type: text/html; charset="utf-8"

<html><body>Click <a href="http://dhl-redelivery.click/pay">here to verify</a></body></html>
--B
Content-Type: application/octet-stream; name="invoice.exe"
Content-Disposition: attachment; filename="invoice.exe"
Content-Transfer-Encoding: base64

QUFB
--B--
"""

QUOTED_PRINTABLE_HTML = b"""From: Apple <x@appleid-secure.xyz>
Subject: Verify
MIME-Version: 1.0
Content-Type: multipart/alternative; boundary="B"

--B
Content-Type: text/html; charset="utf-8"
Content-Transfer-Encoding: quoted-printable

<html><body><a href=3D"http://appleid-verify.xyz/signin?token=3Dabc">Verify=
 now</a></body></html>
--B--
"""

MULTIPART_BOTH_FULL = b"""From: Newsletter <news@techcrunch.com>
Subject: Top stories this week
MIME-Version: 1.0
Content-Type: multipart/alternative; boundary="B"

--B
Content-Type: text/plain; charset="utf-8"

Top stories this week: AI funding rounds and open source model releases.
Read more at https://techcrunch.com/stories and unsubscribe any time.
--B
Content-Type: text/html; charset="utf-8"

<html><body><p>Top stories this week: AI funding rounds and open source model
releases.</p><a href="https://techcrunch.com/stories">Read more</a></body></html>
--B--
"""


# ── URL extraction ───────────────────────────────────────────────────────────

class URLExtractionTests(SimpleTestCase):

    def test_plain_text_urls_still_work(self):
        body, attachments, urls = parse(SINGLE_PART_PLAIN)
        self.assertEqual(urls, ['http://paypa1-verify.xyz/login?id=88'])
        self.assertIn('Unusual activity', body)

    def test_single_part_html(self):
        body, attachments, urls = parse(SINGLE_PART_HTML)
        self.assertEqual(urls, ['http://paypa1-verify.xyz/login?id=88'])

    def test_multipart_html_only(self):
        body, attachments, urls = parse(MULTIPART_HTML_ONLY)
        self.assertEqual(urls, ['http://paypa1-verify.xyz/login'])

    def test_multipart_with_plain_stub_still_finds_html_links(self):
        """The regression that started this: plain stub shadowed the HTML."""
        body, attachments, urls = parse(MULTIPART_PLAIN_STUB)
        self.assertEqual(urls, ['http://dhl-redelivery.click/pay'])

    def test_multipart_mixed_keeps_both_link_and_attachment(self):
        body, attachments, urls = parse(MULTIPART_MIXED_WITH_ATTACHMENT)
        self.assertEqual(urls, ['http://dhl-redelivery.click/pay'])
        self.assertEqual(attachments, ['invoice.exe'])

    def test_quoted_printable_html(self):
        body, attachments, urls = parse(QUOTED_PRINTABLE_HTML)
        self.assertEqual(urls, ['http://appleid-verify.xyz/signin?token=abc'])

    def test_entities_in_query_string_are_unescaped(self):
        html = '<a href="http://evil.example/go?id=1&amp;token=abc">Click</a>'
        self.assertEqual(
            extract_urls_from_html(html),
            ['http://evil.example/go?id=1&token=abc'],
        )

    def test_non_http_schemes_are_ignored(self):
        html = ('<a href="mailto:a@b.com">Mail</a>'
                '<img src="cid:logo123">'
                '<img src="data:image/gif;base64,R0lGOD">'
                '<a href="http://real.example/x">Real</a>')
        self.assertEqual(extract_urls_from_html(html), ['http://real.example/x'])

    def test_unquoted_and_single_quoted_attributes(self):
        html = ("<a href=http://one.example/a>1</a>"
                "<a href='http://two.example/b'>2</a>")
        self.assertEqual(
            extract_urls_from_html(html),
            ['http://one.example/a', 'http://two.example/b'],
        )

    def test_duplicate_links_collapse(self):
        html = ('<a href="http://evil.example/go">Click</a>'
                '<a href="http://evil.example/go">Or here</a>')
        self.assertEqual(extract_urls_from_html(html), ['http://evil.example/go'])

    def test_bare_url_in_html_text_is_found(self):
        html = '<p>Go to http://evil.example/manual now</p>'
        self.assertEqual(extract_urls_from_html(html),
                         ['http://evil.example/manual'])

    def test_empty_input(self):
        self.assertEqual(extract_urls_from_html(''), [])
        self.assertEqual(extract_urls_from_text(''), [])


# ── Body text quality ────────────────────────────────────────────────────────

class BodyTextTests(SimpleTestCase):

    def test_html_tags_are_stripped_from_body(self):
        body, _, _ = parse(SINGLE_PART_HTML)
        self.assertNotIn('<', body)
        self.assertNotIn('href', body)
        self.assertIn('Dear Customer', body)

    def test_stub_plain_part_does_not_win_over_real_html(self):
        """A decoy text/plain alternative must not shadow the rendered HTML."""
        body, _, _ = parse(MULTIPART_PLAIN_STUB)
        self.assertIn('package is on hold', body)
        self.assertNotIn('Please enable HTML', body)

    def test_html_part_is_authoritative_when_both_are_real(self):
        body, _, _ = parse(MULTIPART_BOTH_FULL)
        self.assertIn('Top stories this week', body)

    def test_plain_part_used_when_there_is_no_html(self):
        body, _, _ = parse(SINGLE_PART_PLAIN)
        self.assertIn('Unusual activity was detected', body)

    def test_script_and_style_contents_are_dropped(self):
        raw = (b'Content-Type: text/html; charset="utf-8"\n\n'
               b'<html><head><style>.a{color:red}</style>'
               b'<script>var x=1;</script></head>'
               b'<body>Real content here</body></html>')
        body, _, _ = parse(raw)
        self.assertEqual(body, 'Real content here')

    def test_entities_are_unescaped_in_body(self):
        raw = (b'Content-Type: text/html; charset="utf-8"\n\n'
               b'<html><body>Tom &amp; Jerry&nbsp;are here</body></html>')
        body, _, _ = parse(raw)
        self.assertIn('Tom & Jerry', body)


# ── End-to-end scoring ───────────────────────────────────────────────────────

class ScoringTests(SimpleTestCase):

    def test_html_phishing_gets_url_signal(self):
        """An HTML-only phish must produce a non-zero URL modality score."""
        body, attachments, urls = parse(MULTIPART_PLAIN_STUB)
        result = classify_email(
            sender='no-reply@dhl-delivery.top',
            subject='Package on hold',
            body=body,
            urls=urls,
            attachment='',
        )
        self.assertGreater(result['url_score'], 0.0)
        self.assertEqual(result['extracted_urls'], urls)
        self.assertNotEqual(result['status'], 'safe')

    def test_link_only_phish_is_no_longer_scored_safe(self):
        """
        Bland wording, malice entirely in the link. This is the case that was
        landing as 'safe' with an empty URL list.
        """
        raw = (b'From: IT Support <it-support@sharedocs-cloud.com>\n'
               b'Subject: Shared document\n'
               b'MIME-Version: 1.0\n'
               b'Content-Type: multipart/alternative; boundary="B"\n\n'
               b'--B\nContent-Type: text/plain; charset="utf-8"\n\n'
               b'Please enable HTML.\n'
               b'--B\nContent-Type: text/html; charset="utf-8"\n\n'
               b'<html><body>Hi, a document has been shared with you. '
               b'<a href="http://office365-login.click/auth">View document</a>'
               b'</body></html>\n--B--\n')
        body, attachments, urls = parse(raw)
        self.assertEqual(urls, ['http://office365-login.click/auth'])

        result = classify_email(
            sender='it-support@sharedocs-cloud.com',
            subject='Shared document',
            body=body,
            urls=urls,
        )
        self.assertNotEqual(result['status'], 'safe')
        self.assertIn('.click', result['why_flagged'])

    def test_legitimate_html_newsletter_stays_safe(self):
        body, attachments, urls = parse(MULTIPART_BOTH_FULL)
        result = classify_email(
            sender='news@techcrunch.com',
            subject='Top stories this week',
            body=body,
            urls=urls,
        )
        self.assertEqual(result['status'], 'safe')


# ── Verdict coherence and blind-spot escalation ──────────────────────────────

# A bland email whose only risk signal is the attachment. Deliberately worded to
# score near zero on every trained feature, so any escalation is unambiguous.
BLAND = dict(
    subject='Updated staff handbook',
    body='Please see the attached handbook. Best regards, HR',
)


class VerdictCoherenceTests(SimpleTestCase):
    """
    status is derived from risk_score and never set beside it, so the badge and
    the percentage on the detail page cannot contradict each other. This used to
    be possible: the network path labelled from argmax(probas) while risk_score
    was computed separately from the same probabilities.
    """

    def test_bands_map_as_documented(self):
        self.assertEqual(status_from_risk(RISK_PHISHING), 'phishing')
        self.assertEqual(status_from_risk(RISK_SUSPICIOUS), 'suspicious')
        self.assertEqual(status_from_risk(RISK_SUSPICIOUS - 0.001), 'safe')
        self.assertEqual(status_from_risk(0.0), 'safe')
        self.assertEqual(status_from_risk(1.0), 'phishing')

    def test_status_always_agrees_with_risk_score(self):
        cases = [
            ('hr@somecompany.com',      'handbook.exe'),
            ('billing@dhl-delivery.com','label.exe'),
            ('hr@somecompany.com',      'handbook.pdf'),
            ('billing@dhl-delivery.com',''),
            ('news@techcrunch.com',     ''),
            ('x@paypa1.xyz',            'invoice.scr'),
        ]
        for sender, attachment in cases:
            with self.subTest(sender=sender, attachment=attachment):
                r = classify_email(sender=sender, attachment=attachment, **BLAND)
                self.assertEqual(r['status'], status_from_risk(r['risk_score']))


class BlindSpotEscalationTests(SimpleTestCase):
    """
    The model cannot see attachments at all — there is no attachment feature on
    either side of train/serve (see FeatureParityTests). Until that is a real
    feature and the model is retrained, these floors carry the signal.
    """

    def test_dangerous_attachment_alone_is_suspicious_not_phishing(self):
        """An .exe is dangerous but is not on its own proof of phishing."""
        r = classify_email(sender='hr@somecompany.com',
                           attachment='handbook.exe', **BLAND)
        self.assertEqual(r['status'], 'suspicious')
        self.assertTrue(r['escalated'])
        # The precise incoherence that prompted this: high attachment score
        # sitting next to a safe verdict.
        self.assertGreater(r['attachment_score'], 0.6)
        self.assertEqual(status_from_risk(r['model_risk_score']), 'safe')

    def test_dangerous_attachment_with_impersonation_is_phishing(self):
        r = classify_email(sender='billing@dhl-delivery.com',
                           attachment='label.exe', **BLAND)
        self.assertEqual(r['status'], 'phishing')
        self.assertTrue(r['escalated'])

    def test_benign_attachment_does_not_escalate(self):
        for attachment in ('handbook.pdf', 'notes.docx', ''):
            with self.subTest(attachment=attachment):
                r = classify_email(sender='hr@somecompany.com',
                                   attachment=attachment, **BLAND)
                self.assertFalse(r['escalated'])
                self.assertEqual(r['risk_score'], r['model_risk_score'])

    def test_escalation_never_lowers_a_higher_model_risk(self):
        """The floor asserts 'at least this bad', it does not overwrite."""
        self.assertEqual(_escalate(0.9, 'a@b.com', 'x.exe'), (0.9, []))
        self.assertEqual(_escalate(0.9, 'billing@dhl-delivery.com', 'x.exe'),
                         (0.9, []))
        risk, reasons = _escalate(0.3, 'billing@dhl-delivery.com', 'x.exe')
        self.assertEqual(risk, RISK_PHISHING)
        self.assertTrue(reasons)

    def test_an_override_is_distinguishable_from_a_model_verdict(self):
        """
        An escalated verdict and a model verdict must not read identically on
        the page, or the override is invisible.
        """
        escalated = classify_email(sender='hr@somecompany.com',
                                   attachment='handbook.exe', **BLAND)
        self.assertIn('rule override', escalated['why_flagged'])
        self.assertIn('.exe', escalated['why_flagged'])
        # States what the model said on its own, so the override is auditable.
        self.assertIn('2%', escalated['why_flagged'])

        plain = classify_email(sender='hr@somecompany.com',
                              attachment='handbook.pdf', **BLAND)
        self.assertIn('neural network', plain['why_flagged'])
        self.assertNotIn('rule override', plain['why_flagged'])


# ── Train/serve feature parity ───────────────────────────────────────────────

# What each index means, for readable assertion failures.
FEATURE_NAMES = [
    '0  url present',
    '1  url count',
    '2  suspicious url tld',
    '3  ip literal',
    '4  urgency',
    '5  account threat',
    '6  credential request',
    '7  phishing CTA',
    '8  generic greeting',
    '9  safe signals',
    '10 bulk-mail marker',
    '11 text length',
]

# (name, subject, body, sender, urls)
PARITY_FIXTURES = [
    (
        'phish, url inline in body',
        'Account suspended',
        'Dear Customer your account has been limited click here to verify your '
        'identity within 24 hours http://paypa1-support.net/verify',
        'security@paypa1-support.net',
        ['http://paypa1-support.net/verify'],
    ),
    (
        'phish, url only in href',
        'Shared document',
        'Hi, a document has been shared with you. View document.',
        'it-support@sharedocs-cloud.com',
        ['http://office365-login.click/auth'],
    ),
    (
        'phish, suspicious tld and urgency',
        'Verify now',
        'Urgent: your password will expire. Confirm now or lose access.',
        'alerts@secure-bnk.xyz',
        ['http://password-reset-portal.xyz/update'],
    ),
    (
        'phish, ip literal url',
        'Security alert',
        'Login from new device, verify it was you.',
        'alerts@192-168-alerts.com',
        ['http://192.168.44.10/verify'],
    ),
    (
        'legit newsletter with unsubscribe',
        'Top stories this week',
        'Newsletter top stories this week AI funding rounds and open source '
        'model releases read more, unsubscribe any time.',
        'news@techcrunch.com',
        ['https://techcrunch.com/stories'],
    ),
    (
        'legit business mail, no urls',
        'Meeting notes',
        'Hi team please find attached the meeting notes from yesterday '
        'project sync best regards John',
        'john@company.com',
        [],
    ),
    (
        'brand-impersonating sender, bland body',
        'Invoice attached',
        'Please find the invoice attached. Kind regards, Billing',
        'billing@dhl-delivery.com',
        [],
    ),
    (
        'empty everything',
        '',
        '',
        '',
        [],
    ),
    (
        'url in both body and list, must not double count',
        'Reminder',
        'Go to http://evil.example/go now',
        'x@evil.example',
        ['http://evil.example/go'],
    ),
]


# ── Probes derived from the regexes themselves ───────────────────────────────
# Realistic email fixtures are not enough on their own to catch drift. A regex
# is a list of alternatives, and a fixture only exercises the one it happens to
# contain: adding `suspended` to the index-4 urgency regex is invisible to any
# fixture whose body also says "within 24 hours", because that token already
# made the feature fire. So the probes below are generated from every literal
# in every alternation in BOTH implementations, one probe per token.

PATTERN_LITERAL_RE = re.compile(r"r'([^']*)'")
ALTERNATION_RE = re.compile(r'\(([^)]*)\)')
PLAIN_TOKEN_RE = re.compile(r'[a-z0-9 ]+')


def alternation_tokens(func):
    """Every plain-word literal in every alternation group of func's regexes."""
    tokens = set()
    for pattern in PATTERN_LITERAL_RE.findall(inspect.getsource(func)):
        for group in ALTERNATION_RE.findall(pattern):
            for token in group.split('|'):
                token = token.strip()
                if token and PLAIN_TOKEN_RE.fullmatch(token):
                    tokens.add(token)
    return tokens


def build_probe_texts():
    """One probe per regex token, plus probes for the non-alternation features."""
    probes = []
    for token in sorted(alternation_tokens(serving_features)
                        | alternation_tokens(training_features)):
        probes.append(f'hello {token} world')
        if ' ' not in token:
            # TLD tokens only fire inside a URL
            probes.append(f'see http://evil.{token}/path')
            probes.append(f'see https://evil.{token}/path')
    probes += [
        '',
        'nothing notable in this message at all',
        'http://a.example/x',
        'https://a.example/x',
        ' '.join(f'http://s{i}.example/x' for i in range(8)),   # count saturation
        '1.2.3.4',
        'http://1.2.3.4/verify',
        '999.999.999.999',
        # word-boundary behaviour — these must NOT fire the whole-word regexes
        'suspending expired clicking',
        'unsubscribed newsletters',
    ]
    return probes


PROBE_TEXTS = build_probe_texts()


class FeatureParityTests(SimpleTestCase):
    """
    The saved .pkl was fitted against ml/train_model.py's feature definitions.
    If detector/ml_engine.py computes something different at the same index,
    the network reads the new value as the trained meaning — which is exactly
    how index 10 came to invert the sender-impersonation signal, hidden behind
    a docstring claiming the two agreed.

    These tests are what enforces that agreement now.
    """

    def test_regex_vocabulary_is_identical(self):
        """
        Catches a token added to or removed from either side, naming it,
        before the per-index comparison has to infer it from a value.
        """
        serving = alternation_tokens(serving_features)
        training = alternation_tokens(training_features)
        self.assertEqual(
            serving, training,
            f'regex vocabulary drift — only in ml_engine: '
            f'{sorted(serving - training)}; '
            f'only in train_model: {sorted(training - serving)}',
        )

    def test_every_regex_token_produces_the_same_vector(self):
        """One probe per token, so no alternative can hide behind another."""
        self.assertGreater(len(PROBE_TEXTS), 50)
        for probe in PROBE_TEXTS:
            with self.subTest(probe=probe):
                serving = serving_features('', probe, '', [], '')
                training = training_features(_feature_text('', probe, []))
                for i, (s, t) in enumerate(zip(serving, training)):
                    self.assertEqual(
                        s, t,
                        f'feature drift at index {FEATURE_NAMES[i]} '
                        f'on probe {probe!r}: ml_engine={s} train_model={t}',
                    )

    def test_urls_passed_as_a_list_produce_the_same_vector(self):
        """The same probes again, but via the `urls` list rather than the body."""
        url_probes = [
            [],
            ['http://a.example/x'],
            ['http://evil.xyz/x'],
            ['http://evil.click/x'],
            ['http://1.2.3.4/x'],
            ['https://safe.example/x'],
            [f'http://s{i}.example/x' for i in range(8)],
        ]
        for urls in url_probes:
            with self.subTest(urls=urls):
                serving = serving_features('Subject', 'body text', '', urls, '')
                training = training_features(
                    _feature_text('Subject', 'body text', urls))
                self.assertEqual(serving, training)

    def test_both_extractors_return_twelve_features(self):
        self.assertEqual(len(FEATURE_NAMES), 12)
        self.assertEqual(len(training_features('hello')), 12)
        self.assertEqual(len(serving_features('s', 'b', 'x@y.com', [], '')), 12)

    def test_vectors_match_index_by_index(self):
        for name, subject, body, sender, urls in PARITY_FIXTURES:
            with self.subTest(fixture=name):
                serving = serving_features(subject, body, sender, urls, '')
                training = training_features(_feature_text(subject, body, urls))
                for i, (s, t) in enumerate(zip(serving, training)):
                    self.assertEqual(
                        s, t,
                        f'feature drift at index {FEATURE_NAMES[i]} '
                        f'on fixture {name!r}: ml_engine={s} train_model={t}',
                    )

    # ── The signature difference at indices 0-3 ──────────────────────────────
    # train_model takes one flat text blob with URLs inline; ml_engine takes a
    # separate `urls` list. The intended relationship is that ml_engine folds
    # that list back into the text, so the two agree. These pin it.

    def test_href_only_urls_reach_the_url_features(self):
        """
        A URL recovered from an HTML href never appears in the body. It must
        still set indices 0-3, or HTML phishing is scored with no URL signal.
        """
        subject, body = 'Shared document', 'A document has been shared with you.'
        urls = ['http://office365-login.click/auth']

        without = serving_features(subject, body, 'a@b.com', [], '')
        with_url = serving_features(subject, body, 'a@b.com', urls, '')

        self.assertEqual(without[0:3], [0, 0.0, 0])
        self.assertEqual(with_url[0], 1)          # url present
        self.assertEqual(with_url[1], 1 / 5)      # one url
        self.assertEqual(with_url[2], 1)          # .click is a suspicious tld

        # And the folded text is what train_model would have been given
        self.assertIn(urls[0], _feature_text(subject, body, urls))

    def test_inline_urls_are_not_double_counted(self):
        """A URL in both the body and the list is one URL, not two."""
        subject, body = 'Reminder', 'Go to http://evil.example/go now'
        urls = ['http://evil.example/go']
        self.assertEqual(serving_features(subject, body, 'a@b.com', urls, '')[1],
                         1 / 5)
        self.assertEqual(_feature_text(subject, body, urls).count('evil.example'), 1)

    def test_url_count_saturates_the_same_way_on_both_sides(self):
        urls = [f'http://evil{i}.example/go' for i in range(8)]
        serving = serving_features('s', 'body', 'a@b.com', urls, '')
        training = training_features(_feature_text('s', 'body', urls))
        self.assertEqual(serving[1], 1.0)
        self.assertEqual(serving[1], training[1])

    # ── The specific regression ──────────────────────────────────────────────

    def test_index_10_is_the_bulk_mail_marker_not_a_sender_signal(self):
        """
        Index 10 was trained as unsubscribe/newsletter/view-in-browser — a
        *safe* marker. Feeding sender-domain impersonation into it inverted
        the signal: firing it pushed the verdict towards safe.
        """
        # Brand-impersonating sender on a suspicious TLD, no bulk-mail wording
        impersonating = serving_features(
            'Security notice', 'Please review your account details.',
            'service@paypa1-secure.xyz', ['http://paypa1-secure.xyz/review'], '',
        )
        self.assertEqual(impersonating[10], 0,
                         'index 10 must not carry the sender-domain signal')

        # Genuine bulk mail
        newsletter = serving_features(
            'Weekly digest', 'Read more below, unsubscribe any time.',
            'news@techcrunch.com', [], '',
        )
        self.assertEqual(newsletter[10], 1)

    def test_sender_is_not_used_by_the_feature_vector(self):
        """
        Documents a known blind spot: the model sees nothing about the sender.
        If this ever starts failing, a sender feature was added on one side —
        it must be added to train_model.py and retrained, not just here.
        """
        base = serving_features('Invoice', 'Please find the invoice.', '', [], '')
        for sender in ('billing@dhl-delivery.com', 'x@paypa1.xyz', 'a@gmail.com'):
            with self.subTest(sender=sender):
                self.assertEqual(
                    serving_features('Invoice', 'Please find the invoice.',
                                     sender, [], ''),
                    base,
                )

    def test_attachment_is_not_used_by_the_feature_vector(self):
        """
        Same, for attachments: _score_attachment() returns 0.92 for a .exe and
        the UI prints a warning, but the model's verdict is unaffected.
        """
        base = serving_features('Invoice', 'Please find the invoice.',
                                'a@b.com', [], '')
        for attachment in ('invoice.exe', 'notes.pdf', 'payload.scr'):
            with self.subTest(attachment=attachment):
                self.assertEqual(
                    serving_features('Invoice', 'Please find the invoice.',
                                     'a@b.com', [], attachment),
                    base,
                )


# ── Backfill of emails synced before the URL fix ─────────────────────────────

BACKFILL_FIXTURE = b"""From: DHL <no-reply@dhl-delivery.top>
Subject: Package on hold
Message-ID: <abc123@dhl-delivery.top>
Date: Mon, 20 Jul 2026 09:15:00 +0100
MIME-Version: 1.0
Content-Type: multipart/alternative; boundary="B"

--B
Content-Type: text/plain; charset="utf-8"

Please enable HTML to view this message.
--B
Content-Type: text/html; charset="utf-8"

<html><body><p>Dear User, your package is on hold. Confirm now to release it:
<a href="http://dhl-redelivery.click/pay">Update now</a></p></body></html>
--B--
"""


class FakeConnector:
    """
    Stands in for GmailIMAPConnector during backfill tests.

    Serves the fixture keyed by Message-ID, in the shape the real
    fetch_by_message_ids() returns: {message_id: (parsed, folder_label)}.
    """
    served = [BACKFILL_FIXTURE]
    label = 'All Mail'
    calls = []

    def __init__(self, email_address, app_password):
        pass

    def fetch_by_message_ids(self, message_ids, progress=None):
        FakeConnector.calls.append(set(message_ids))
        available = {}
        for raw in FakeConnector.served:
            parsed = GmailIMAPConnector.parse_message(raw)
            key = normalize_message_id(parsed['message_id'])
            available[key] = (parsed, FakeConnector.label)
        wanted = {normalize_message_id(m) for m in message_ids}
        return True, {k: v for k, v in available.items() if k in wanted}


class ParseMessageTests(SimpleTestCase):
    """
    parse_message() is shared by fetch_emails() and fetch_by_message_ids(),
    so a re-fetch yields exactly what the original sync would have.
    """

    def test_parses_headers_body_and_urls(self):
        parsed = GmailIMAPConnector.parse_message(BACKFILL_FIXTURE)
        self.assertEqual(parsed['sender'], 'no-reply@dhl-delivery.top')
        self.assertEqual(parsed['subject'], 'Package on hold')
        self.assertEqual(parsed['message_id'], '<abc123@dhl-delivery.top>')
        self.assertEqual(parsed['urls'], ['http://dhl-redelivery.click/pay'])
        self.assertIn('package is on hold', parsed['body'])
        # The decoy plain part must not be what gets stored
        self.assertNotIn('enable HTML', parsed['body'])

    def test_message_id_normalisation_survives_folding_and_padding(self):
        self.assertEqual(
            normalize_message_id('  <abc@x.com>\r\n '),
            normalize_message_id('<abc@x.com>'),
        )


ARCHIVED_FIXTURE = BACKFILL_FIXTURE.replace(
    b'<abc123@dhl-delivery.top>', b'<archived456@dhl-delivery.top>'
)


class FakeIMAPServer:
    """
    A scripted IMAP server, standing in for imaplib.IMAP4_SSL.

    Covers the protocol interaction fetch_by_message_ids() actually performs —
    LIST, SELECT, SEARCH HEADER, FETCH — which the command-level fake bypasses.
    """

    ATTRS = {
        '[Gmail]/All Mail': rb'\HasNoChildren \All',
        '[Gmail]/Spam':     rb'\HasNoChildren \Junk',
        '[Gmail]/Trash':    rb'\HasNoChildren \Trash',
        'INBOX':            rb'\HasNoChildren',
    }

    def __init__(self, folders):
        self.folders = folders          # {folder_name: [raw_message_bytes]}
        self.selected = None
        self.searches = []              # (folder, criteria) per SEARCH
        self.selects = []
        self.readonly_selects = []
        self.logged_in = False

    def login(self, user, password):
        self.logged_in = True
        return 'OK', [b'authenticated']

    def list(self):
        lines = [
            b'(' + self.ATTRS.get(name, rb'\HasNoChildren') + b') "/" "'
            + name.encode() + b'"'
            for name in self.folders
        ]
        return 'OK', lines

    def select(self, folder, readonly=False):
        self.selects.append(folder)
        if folder not in self.folders:
            return 'NO', [b'Unknown Mailbox']
        self.selected = folder
        if readonly:
            self.readonly_selects.append(folder)
        return 'OK', [str(len(self.folders[folder])).encode()]

    def search(self, charset, *criteria):
        self.searches.append((self.selected, criteria))
        if criteria[0] != 'HEADER' or criteria[1] != 'Message-ID':
            raise AssertionError(f'unexpected SEARCH criteria: {criteria}')
        quoted = criteria[2]
        if not (quoted.startswith('"') and quoted.endswith('"')):
            raise AssertionError(f'Message-ID was not quoted: {quoted!r}')
        wanted = quoted[1:-1]

        hits = []
        for index, raw in enumerate(self.folders[self.selected], start=1):
            parsed = email.message_from_bytes(raw)
            if normalize_message_id(parsed.get('Message-ID', '')) == wanted:
                hits.append(str(index).encode())
        return 'OK', [b' '.join(hits)]

    def fetch(self, seq, spec):
        if spec != '(BODY.PEEK[])':
            raise AssertionError(f'fetch must peek, not mark read: {spec}')
        raw = self.folders[self.selected][int(seq) - 1]
        return 'OK', [(b'%s (BODY[] {%d}' % (seq, len(raw)), raw), b')']

    def close(self):
        self.selected = None

    def logout(self):
        self.logged_in = False


class FetchByMessageIdTests(SimpleTestCase):
    """
    The lookup path itself: does it speak the protocol correctly, does it find
    mail that is no longer in the Inbox, and does its cost track the number of
    rows being repaired rather than the size of the mailbox?
    """

    def run_lookup(self, folders, message_ids):
        server = FakeIMAPServer(folders)
        connector = GmailIMAPConnector('u@gmail.com', 'pw')
        with mock.patch('detector.imap_connector.imaplib.IMAP4_SSL',
                        return_value=server) as ctor:
            ok, result = connector.fetch_by_message_ids(message_ids)
        return server, ok, result, ctor

    def test_finds_a_message_that_was_archived_out_of_the_inbox(self):
        """
        The reported symptom: mail archived after syncing lives only in All
        Mail. Searching Inbox and Spam alone would report it unrecoverable.
        """
        server, ok, result, _ = self.run_lookup(
            {'INBOX': [], '[Gmail]/All Mail': [ARCHIVED_FIXTURE]},
            ['<archived456@dhl-delivery.top>'],
        )
        self.assertTrue(ok)
        parsed, label = result['<archived456@dhl-delivery.top>']
        self.assertEqual(label, 'All Mail')
        self.assertEqual(parsed['urls'], ['http://dhl-redelivery.click/pay'])

    def test_finds_a_message_in_spam(self):
        server, ok, result, _ = self.run_lookup(
            {'[Gmail]/All Mail': [], '[Gmail]/Spam': [BACKFILL_FIXTURE]},
            ['<abc123@dhl-delivery.top>'],
        )
        self.assertTrue(ok)
        self.assertEqual(result['<abc123@dhl-delivery.top>'][1], 'Spam')

    def test_cost_scales_with_rows_repaired_not_mailbox_size(self):
        """
        The whole point of SEARCH over a header scan: two rows to repair costs
        two searches whether All Mail holds three messages or fifty thousand.
        """
        bulk = [
            BACKFILL_FIXTURE.replace(b'<abc123@', b'<filler%d@' % i)
            for i in range(200)
        ]
        server, ok, result, _ = self.run_lookup(
            {'[Gmail]/All Mail': bulk + [BACKFILL_FIXTURE, ARCHIVED_FIXTURE]},
            ['<abc123@dhl-delivery.top>', '<archived456@dhl-delivery.top>'],
        )
        self.assertTrue(ok)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(server.searches), 2)

    def test_a_found_message_is_not_searched_for_again_in_later_folders(self):
        server, ok, result, _ = self.run_lookup(
            {'[Gmail]/All Mail': [BACKFILL_FIXTURE],
             '[Gmail]/Spam': [],
             '[Gmail]/Trash': []},
            ['<abc123@dhl-delivery.top>'],
        )
        self.assertEqual(len(server.searches), 1)
        self.assertEqual(server.searches[0][0], '[Gmail]/All Mail')

    def test_folders_are_opened_read_only(self):
        """A repair must never mark the user's mail as read."""
        server, ok, result, _ = self.run_lookup(
            {'[Gmail]/All Mail': [BACKFILL_FIXTURE]},
            ['<abc123@dhl-delivery.top>'],
        )
        self.assertEqual(server.readonly_selects, ['[Gmail]/All Mail'])
        self.assertEqual(server.selects, server.readonly_selects)

    def test_an_absent_folder_does_not_abort_the_lookup(self):
        """Accounts without a Trash or Spam folder must still be repairable."""
        server, ok, result, _ = self.run_lookup(
            {'[Gmail]/All Mail': [BACKFILL_FIXTURE]},
            ['<abc123@dhl-delivery.top>'],
        )
        self.assertTrue(ok)
        self.assertEqual(len(result), 1)

    def test_a_message_that_is_gone_returns_no_result_not_an_error(self):
        server, ok, result, _ = self.run_lookup(
            {'[Gmail]/All Mail': []},
            ['<deleted@dhl-delivery.top>'],
        )
        self.assertTrue(ok)
        self.assertEqual(result, {})

    def test_a_connection_timeout_is_reported_not_raised(self):
        connector = GmailIMAPConnector('u@gmail.com', 'pw')
        with mock.patch('detector.imap_connector.imaplib.IMAP4_SSL',
                        side_effect=TimeoutError('timed out')):
            ok, result = connector.fetch_by_message_ids(['<a@b.com>'])
        self.assertFalse(ok)
        self.assertIn('timed out', result)

    def test_a_socket_timeout_is_configured(self):
        """
        imaplib defaults to no timeout, which would let an unattended backfill
        block forever on a stalled read.
        """
        _, _, _, ctor = self.run_lookup(
            {'[Gmail]/All Mail': [BACKFILL_FIXTURE]},
            ['<abc123@dhl-delivery.top>'],
        )
        timeout = ctor.call_args.kwargs.get('timeout')
        self.assertIsNotNone(timeout, 'no socket timeout was passed to imaplib')
        self.assertGreater(timeout, 0)

    def test_no_message_ids_makes_no_connection(self):
        connector = GmailIMAPConnector('u@gmail.com', 'pw')
        with mock.patch('detector.imap_connector.imaplib.IMAP4_SSL') as ctor:
            ok, result = connector.fetch_by_message_ids([])
        self.assertTrue(ok)
        self.assertEqual(result, {})
        ctor.assert_not_called()


class FakeListConn:
    """Just enough of an IMAP connection to answer LIST."""

    def __init__(self, lines, status='OK'):
        self.lines = lines
        self.status = status

    def list(self):
        return self.status, self.lines


# Gmail's real LIST output, abridged.
GMAIL_LIST = [
    b'(\\HasNoChildren) "/" "INBOX"',
    b'(\\HasChildren \\Noselect) "/" "[Gmail]"',
    b'(\\HasNoChildren \\All) "/" "[Gmail]/All Mail"',
    b'(\\HasNoChildren \\Junk) "/" "[Gmail]/Spam"',
    b'(\\HasNoChildren \\Trash) "/" "[Gmail]/Trash"',
    b'(\\HasNoChildren \\Sent) "/" "[Gmail]/Sent Mail"',
]

# The same account with the interface language set to French. The display
# names are translated; the special-use attributes are not.
GMAIL_LIST_FRENCH = [
    b'(\\HasNoChildren) "/" "INBOX"',
    b'(\\HasNoChildren \\All) "/" "[Gmail]/Tous les messages"',
    b'(\\HasNoChildren \\Junk) "/" "[Gmail]/Spam"',
    b'(\\HasNoChildren \\Trash) "/" "[Gmail]/Corbeille"',
]


class FolderResolutionTests(SimpleTestCase):
    """
    A backfill has to look in All Mail, because archiving a message removes it
    from the Inbox. Its name is localised per account, so it is located by the
    \\All special-use attribute rather than by the English string.
    """

    def setUp(self):
        self.connector = GmailIMAPConnector('u@gmail.com', 'pw')

    def test_gmail_special_use_folders_are_resolved(self):
        folders = self.connector._resolve_folders(FakeListConn(GMAIL_LIST))
        self.assertEqual(folders, [
            ('[Gmail]/All Mail', 'All Mail'),
            ('[Gmail]/Spam', 'Spam'),
            ('[Gmail]/Trash', 'Trash'),
        ])

    def test_localised_folder_names_are_resolved_by_attribute(self):
        """The English fallback would silently find nothing on this account."""
        folders = self.connector._resolve_folders(FakeListConn(GMAIL_LIST_FRENCH))
        names = [name for name, _ in folders]
        self.assertIn('[Gmail]/Tous les messages', names)
        self.assertIn('[Gmail]/Corbeille', names)

    def test_all_mail_is_searched_before_spam_and_trash(self):
        """
        All Mail holds everything that is not spam or trash, so trying it
        first resolves the common case in one pass.
        """
        folders = self.connector._resolve_folders(FakeListConn(GMAIL_LIST))
        self.assertEqual(folders[0][1], 'All Mail')

    def test_a_server_without_special_use_falls_back_to_inbox(self):
        """Non-Gmail IMAP may advertise no \\All folder at all."""
        folders = self.connector._resolve_folders(
            FakeListConn([b'(\\HasNoChildren) "/" "INBOX"'])
        )
        self.assertIn(('INBOX', 'Inbox'), folders)

    def test_a_failed_list_still_yields_the_default_folders(self):
        folders = self.connector._resolve_folders(FakeListConn(None, status='NO'))
        self.assertIn(('[Gmail]/All Mail', 'All Mail'), folders)
        self.assertIn(('INBOX', 'Inbox'), folders)

    def test_search_arguments_are_quoted(self):
        """An unquoted Message-ID would be a syntax error to the server."""
        self.assertEqual(
            GmailIMAPConnector._quote('<abc@mail.gmail.com>'),
            '"<abc@mail.gmail.com>"',
        )
        self.assertEqual(
            GmailIMAPConnector._quote('a"b'),
            '"a\\"b"',
        )


class BackfillCommandTests(TestCase):
    """
    A re-sync cannot fix these rows: sync_gmail_account() skips any message_id
    already stored, and the saved body is post-flattening text with the hrefs
    already deleted. The command re-fetches from Gmail instead.
    """

    def setUp(self):
        FakeConnector.served = [BACKFILL_FIXTURE]
        FakeConnector.label = 'All Mail'
        FakeConnector.calls = []
        patcher = mock.patch(
            'detector.management.commands.backfill_urls.GmailIMAPConnector',
            FakeConnector,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        self.account = GmailAccount(email_address='user@gmail.com')
        self.account.app_password = 'abcd efgh ijkl mnop'
        self.account.save()

    def make_email(self, **overrides):
        """
        A row as the pre-fix sync would have written it: flattened body, no
        URLs, and a verdict reached without any URL signal.
        """
        defaults = dict(
            account=self.account,
            sender='no-reply@dhl-delivery.top',
            sender_domain='dhl-delivery.top',
            subject='Package on hold',
            body='Dear User, your package is on hold. Confirm now to release it: Update now',
            message_id='<abc123@dhl-delivery.top>',
            status='safe',
            risk_score=0.1,
            url_score=0.0,
            extracted_urls=[],
        )
        defaults.update(overrides)
        return Email.objects.create(**defaults)

    def run_command(self, *args):
        out = StringIO()
        call_command('backfill_urls', *args, stdout=out, stderr=out)
        return out.getvalue()

    def test_backfills_links_onto_an_existing_row(self):
        email_obj = self.make_email()
        self.run_command()
        email_obj.refresh_from_db()

        self.assertEqual(email_obj.extracted_urls,
                         ['http://dhl-redelivery.click/pay'])
        # The stored body becomes the HTML alternative, not the decoy
        self.assertNotIn('enable HTML', email_obj.body)

    def test_rescoring_uses_the_recovered_urls(self):
        """
        The old scores were computed as though the email had no links, so
        leaving them would show phishing links beside a no-URL verdict.
        """
        email_obj = self.make_email()
        self.run_command()
        email_obj.refresh_from_db()

        self.assertGreater(email_obj.url_score, 0.0)

    def test_a_verdict_that_becomes_a_threat_raises_an_alert(self):
        """The original sync logged nothing, because it scored the mail safe."""
        email_obj = self.make_email()
        self.run_command()
        email_obj.refresh_from_db()

        logs = ScanLog.objects.filter(email=email_obj)
        if email_obj.status in ('phishing', 'suspicious'):
            self.assertEqual(logs.count(), 1)
            self.assertIn('re-scanned', logs.first().message)
        else:
            self.assertEqual(logs.count(), 0)

    def test_urls_only_leaves_the_existing_verdict_alone(self):
        email_obj = self.make_email()
        self.run_command('--urls-only')
        email_obj.refresh_from_db()

        self.assertEqual(email_obj.extracted_urls,
                         ['http://dhl-redelivery.click/pay'])
        self.assertEqual(email_obj.status, 'safe')
        self.assertEqual(email_obj.risk_score, 0.1)
        self.assertEqual(ScanLog.objects.count(), 0)

    def test_dry_run_writes_nothing(self):
        email_obj = self.make_email()
        output = self.run_command('--dry-run')
        email_obj.refresh_from_db()

        self.assertEqual(email_obj.extracted_urls, [])
        self.assertEqual(email_obj.status, 'safe')
        self.assertIn('would repair 1', output)

    def test_rows_that_already_have_links_are_left_alone(self):
        """
        Post-fix emails are already correct — the default pass skips them, so
        nothing is re-fetched and no verdict moves underneath the user.
        """
        self.make_email(extracted_urls=['http://dhl-redelivery.click/pay'])
        self.run_command()

        self.assertEqual(FakeConnector.calls, [])

    def test_all_flag_reprocesses_rows_that_already_have_links(self):
        """
        A decoy plain part could leave a row with a URL but the wrong body,
        which the default filter has no way to detect.
        """
        email_obj = self.make_email(extracted_urls=['http://old.example'],
                                    body='Please enable HTML to view this message.')
        self.run_command('--all')
        email_obj.refresh_from_db()

        self.assertEqual(email_obj.extracted_urls,
                         ['http://dhl-redelivery.click/pay'])
        self.assertNotIn('enable HTML', email_obj.body)

    def test_an_email_no_longer_in_gmail_keeps_its_data(self):
        """
        Deleted or archived mail cannot be re-fetched; the row must survive
        untouched rather than being blanked.
        """
        FakeConnector.served = []
        email_obj = self.make_email()
        output = self.run_command()
        email_obj.refresh_from_db()

        self.assertEqual(email_obj.extracted_urls, [])
        self.assertEqual(email_obj.status, 'safe')
        self.assertIn('could not be found in Gmail', output)

    def test_a_row_with_no_message_id_is_reported_not_crashed(self):
        """Seeded demo emails have no Message-ID and cannot be matched back."""
        email_obj = self.make_email(message_id='')
        output = self.run_command()
        email_obj.refresh_from_db()

        self.assertEqual(email_obj.extracted_urls, [])
        self.assertIn('no Message-ID', output)

    def test_the_folder_the_sync_recorded_is_preserved(self):
        """
        why_flagged carries a '[Found in Spam]' tag from the original sync.
        The backfill finds the message in All Mail, which says nothing about
        where it was delivered, so re-deriving the tag would lose that.
        """
        email_obj = self.make_email(
            why_flagged='[Found in Spam] Suspicious sender domain'
        )
        self.run_command()
        email_obj.refresh_from_db()

        self.assertTrue(email_obj.why_flagged.startswith('[Found in Spam] '))
        self.assertNotIn('All Mail', email_obj.why_flagged)

    def test_no_folder_tag_is_invented_for_mail_that_had_none(self):
        email_obj = self.make_email(why_flagged='Nothing suspicious found')
        self.run_command()
        email_obj.refresh_from_db()

        self.assertNotIn('[Found in', email_obj.why_flagged)

    def test_unknown_account_is_rejected(self):
        self.make_email()
        output = self.run_command('--account', 'nobody@gmail.com')

        self.assertIn('No active account', output)
        self.assertEqual(FakeConnector.calls, [])


# ── Re-scoring stored rows onto the current engine ───────────────────────────

class RescoreCommandTests(TestCase):
    """
    A stored verdict was produced by whatever the engine looked like when the
    row was written, so an engine change leaves the table holding two regimes.
    rescore_emails brings old rows forward from data already in the database —
    no Gmail access, and it reaches rows backfill_urls cannot, because those are
    selected by account.
    """

    # Bland wording, risk only in the attachment: the model scores this near
    # zero, so the escalation floor is what moves it.
    EXE_ROW = dict(
        attachment_name='handbook.exe',
        has_attachment=True,
    )

    def make_email(self, **overrides):
        defaults = dict(
            sender='hr@somecompany.com',
            subject='Updated staff handbook',
            body='Please see the attached handbook. Best regards, HR',
            status='safe',
            risk_score=0.02,
            extracted_urls=[],
        )
        defaults.update(overrides)
        return Email.objects.create(**defaults)

    def run_command(self, *args):
        out = StringIO()
        call_command('rescore_emails', *args, stdout=out, stderr=out)
        return out.getvalue()

    # ── Reaches rows backfill_urls cannot ────────────────────────────────────

    def test_rescores_a_seeded_row_with_no_account_or_message_id(self):
        """
        backfill_urls filters on account=<account>, so it selects none of these.
        This command must not need an account, a Message-ID, or the network.
        """
        email_obj = self.make_email(**self.EXE_ROW)
        self.assertIsNone(email_obj.account)
        self.assertEqual(email_obj.message_id, '')

        self.run_command()
        email_obj.refresh_from_db()
        self.assertEqual(email_obj.status, 'suspicious')

    def test_is_idempotent(self):
        email_obj = self.make_email(**self.EXE_ROW)
        self.run_command()
        email_obj.refresh_from_db()
        first = (email_obj.status, email_obj.risk_score, email_obj.why_flagged)

        self.run_command()
        email_obj.refresh_from_db()
        self.assertEqual(
            (email_obj.status, email_obj.risk_score, email_obj.why_flagged),
            first,
        )

    # ── Direction asymmetry ──────────────────────────────────────────────────

    def test_escalations_are_applied_by_default(self):
        email_obj = self.make_email(**self.EXE_ROW)
        output = self.run_command()
        email_obj.refresh_from_db()

        self.assertEqual(email_obj.status, 'suspicious')
        self.assertIn('Escalations (applied)', output)

    def test_de_escalations_are_listed_but_not_applied_by_default(self):
        """
        Lowering a verdict un-flags mail someone may already have acted on, so a
        default run reports it and leaves the row as it is.
        """
        email_obj = self.make_email(status='phishing', risk_score=0.91)
        output = self.run_command()
        email_obj.refresh_from_db()

        self.assertEqual(email_obj.status, 'phishing')
        self.assertEqual(email_obj.risk_score, 0.91)
        self.assertIn('De-escalations (NOT applied)', output)
        self.assertIn('--apply-de-escalations', output)

    def test_de_escalations_are_applied_only_with_the_flag(self):
        email_obj = self.make_email(status='phishing', risk_score=0.91)
        self.run_command('--apply-de-escalations')
        email_obj.refresh_from_db()

        self.assertEqual(email_obj.status, 'safe')
        self.assertLess(email_obj.risk_score, RISK_SUSPICIOUS)

    def test_a_default_run_never_lowers_any_verdict(self):
        rows = [
            self.make_email(status='phishing', risk_score=0.91),
            self.make_email(status='suspicious', risk_score=0.44),
            self.make_email(**self.EXE_ROW),
        ]
        before = [(e.id, e.status) for e in rows]
        self.run_command()

        for email_id, old in before:
            with self.subTest(email_id=email_id):
                new = Email.objects.get(id=email_id).status
                self.assertGreaterEqual(SEVERITY[new], SEVERITY[old])

    # ── Refusal when the model is missing ────────────────────────────────────

    def test_refuses_to_run_without_the_trained_model(self):
        """
        Re-scoring through the rule-based fallback would swap model verdicts for
        rule verdicts on every row, with nothing in the data recording it.
        """
        email_obj = self.make_email(status='phishing', risk_score=0.91)
        with mock.patch(
            'detector.management.commands.rescore_emails._load_model',
            return_value=False,
        ):
            with self.assertRaises(CommandError) as ctx:
                self.run_command()

        self.assertIn('rule-based fallback', str(ctx.exception))
        email_obj.refresh_from_db()
        self.assertEqual(email_obj.status, 'phishing')

    # ── Bookkeeping ──────────────────────────────────────────────────────────

    def test_dry_run_writes_nothing(self):
        email_obj = self.make_email(**self.EXE_ROW)
        output = self.run_command('--dry-run')
        email_obj.refresh_from_db()

        self.assertEqual(email_obj.status, 'safe')
        self.assertEqual(email_obj.risk_score, 0.02)
        self.assertIn('dry-run', output)
        self.assertEqual(ScanLog.objects.count(), 0)

    def test_an_escalation_raises_an_alert(self):
        """The original sync never alerted on this row; the escalation must."""
        self.make_email(**self.EXE_ROW)
        self.run_command()

        log = ScanLog.objects.get()
        self.assertEqual(log.level, 'warning')
        self.assertIn('rescore_emails', log.message)

    def test_the_folder_tag_the_sync_recorded_is_preserved(self):
        email_obj = self.make_email(
            why_flagged='[Found in Spam] No suspicious indicators detected.',
            **self.EXE_ROW
        )
        self.run_command()
        email_obj.refresh_from_db()

        self.assertTrue(email_obj.why_flagged.startswith('[Found in Spam] '))
        self.assertIn('rule override', email_obj.why_flagged)

    def test_no_folder_tag_is_invented_for_mail_that_had_none(self):
        email_obj = self.make_email(why_flagged='', **self.EXE_ROW)
        self.run_command()
        email_obj.refresh_from_db()

        self.assertNotIn('[Found in', email_obj.why_flagged)

    def test_scores_are_refreshed_even_when_the_verdict_does_not_move(self):
        """
        Drift can move the numbers without crossing a band. Leaving stale scores
        beside a current verdict is the incoherence this work set out to remove.
        """
        email_obj = self.make_email(attachment_name='notes.pdf',
                                    has_attachment=True,
                                    attachment_score=0.99)
        self.run_command()
        email_obj.refresh_from_db()

        self.assertEqual(email_obj.status, 'safe')
        self.assertLess(email_obj.attachment_score, 0.5)

    def test_unknown_account_is_rejected(self):
        self.make_email()
        with self.assertRaises(CommandError) as ctx:
            self.run_command('--account', 'nobody@gmail.com')
        self.assertIn('No emails found', str(ctx.exception))


# ── Console output under a redirected Windows console ─────────────────────────

def codepage_stream(encoding):
    """
    A stdout stand-in that behaves like a redirected Windows console.

    Python uses UTF-8 for an *attached* console (PEP 528, io._WindowsConsoleIO)
    but falls back to the locale encoding the moment stdout is piped or
    redirected to a file, which is what happens under CI, a task scheduler, or
    `cmd > log.txt`. StringIO accepts any codepoint and so hides this entirely.
    errors='strict' is the point: it raises exactly as the real stream does.
    """
    buf = io.BytesIO()
    return buf, io.TextIOWrapper(buf, encoding=encoding, errors='strict',
                                 newline='')


class ConsoleEncodingTests(TestCase):
    """
    Non-ASCII in console output crashes these commands when stdout is
    redirected. Not hypothetically: backfill_urls wrote a U+2500 box rule per
    account before any verdict changed, so the command could not complete a
    single run under redirection.

    Which codepoints raise depends on the codepage, so asserting ASCII is the
    only portable rule. U+2014 EM DASH happens to be encodable in cp1252 and so
    survived there, but fails on cp437/cp850; U+2500 fails on cp1252 and
    succeeds on cp437; U+2192 fails on all of them.
    """

    CODEPAGES = ('cp1252', 'cp437', 'cp850', 'ascii')

    def run_on_codepage(self, command, encoding, *args):
        buf, stream = codepage_stream(encoding)
        call_command(command, *args, stdout=stream, stderr=stream)
        stream.flush()
        return buf.getvalue().decode(encoding)

    def setUp(self):
        FakeConnector.served = [BACKFILL_FIXTURE]
        FakeConnector.label = 'All Mail'
        FakeConnector.calls = []
        patcher = mock.patch(
            'detector.management.commands.backfill_urls.GmailIMAPConnector',
            FakeConnector,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        self.account = GmailAccount(email_address='user@gmail.com')
        self.account.app_password = 'abcd efgh ijkl mnop'
        self.account.save()

    # ── Interpolated data, not literals ──────────────────────────────────────

    # A real stored subject from db.sqlite3 (row 10). U+2014 is encodable in
    # cp1252 as byte 0x97 but not in cp437/cp850/ascii, so it exercises both
    # sides of the degradation.
    NON_ASCII_SUBJECT = 'Remote job offer — $3,000/month work from home'

    def test_a_non_ascii_subject_degrades_instead_of_raising(self):
        """
        Subjects are attacker-supplied and never guaranteed ASCII, so the fix
        cannot be "make every literal ASCII". handle() relaxes the stream's
        error handler, which covers every interpolated field at once and leaves
        the stored data alone — only the rendering degrades.
        """
        for encoding in self.CODEPAGES:
            with self.subTest(encoding=encoding):
                Email.objects.all().delete()
                stored = Email.objects.create(
                    sender='hr@remote-jobs-worldwide.net',
                    subject=self.NON_ASCII_SUBJECT,
                    body='We found your profile online.',
                    status='phishing', risk_score=0.91, extracted_urls=[],
                )
                out = self.run_on_codepage('rescore_emails', encoding)

                # The row is reported, and the run completed.
                self.assertIn('De-escalations', out)
                self.assertIn('Remote job offer', out)

                if encoding == 'cp1252':
                    # Encodable here: full fidelity is preserved.
                    self.assertIn('—', out)
                else:
                    # Not encodable: one character degrades, nothing raises.
                    self.assertIn('?', out)
                    self.assertNotIn('—', out)

                # The stored subject is untouched either way.
                stored.refresh_from_db()
                self.assertEqual(stored.subject, self.NON_ASCII_SUBJECT)

    def test_backfill_output_is_encodable_on_every_codepage(self):
        """
        Covers the per-account header, which fires unconditionally, and the
        old_status -> new_status transition line.

        Now that handle() relaxes the error handler this asserts the command
        completes rather than that every byte was encodable; the ASCII rule for
        our own literals is enforced statically below.
        """
        for encoding in self.CODEPAGES:
            with self.subTest(encoding=encoding):
                Email.objects.all().delete()
                Email.objects.create(
                    account=self.account,
                    sender='no-reply@dhl-delivery.top',
                    subject='Package on hold',
                    body='Dear User, confirm now to release it.',
                    message_id='<abc123@dhl-delivery.top>',
                    status='safe',
                    risk_score=0.1,
                    extracted_urls=[],
                )
                out = self.run_on_codepage('backfill_urls', encoding)
                self.assertIn('user@gmail.com', out)

    def test_rescore_output_is_encodable_on_every_codepage(self):
        """Covers both the escalation and the de-escalation report paths."""
        for encoding in self.CODEPAGES:
            with self.subTest(encoding=encoding):
                Email.objects.all().delete()
                Email.objects.create(          # escalates
                    sender='hr@somecompany.com',
                    subject='Updated staff handbook',
                    body='Please see the attached handbook. Best regards, HR',
                    attachment_name='handbook.exe', has_attachment=True,
                    status='safe', risk_score=0.02, extracted_urls=[],
                )
                Email.objects.create(          # de-escalates
                    sender='hr@somecompany.com',
                    subject='Updated staff handbook',
                    body='Please see the attached handbook. Best regards, HR',
                    status='phishing', risk_score=0.91, extracted_urls=[],
                )
                out = self.run_on_codepage('rescore_emails', encoding)
                self.assertIn('De-escalations', out)
                self.assertIn('Escalations', out)

    def test_every_console_literal_in_these_commands_is_ascii(self):
        """
        A static guard, because the crashing site only executes when an account
        exists and a run that finds no accounts skips it. Docstrings and DB text
        are excluded: they are never encoded to the console codepage. seeded
        email bodies legitimately contain non-ASCII and are not console output.
        """
        import ast
        import pathlib

        for name in ('backfill_urls.py', 'rescore_emails.py'):
            path = (pathlib.Path(__file__).parent / 'management' / 'commands'
                    / name)
            tree = ast.parse(path.read_text(encoding='utf-8'))

            # Collect literals that flow into self.stdout/self.stderr writes or
            # a CommandError, i.e. everything that reaches the console.
            offenders = []
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                target = ''
                if isinstance(node.func, ast.Attribute):
                    target = node.func.attr
                elif isinstance(node.func, ast.Name):
                    target = node.func.id
                if target not in ('write', 'CommandError'):
                    continue
                for sub in ast.walk(node):
                    if (isinstance(sub, ast.Constant)
                            and isinstance(sub.value, str)
                            and not sub.value.isascii()):
                        offenders.append(
                            f'{name}:{sub.lineno} '
                            f'{ascii(sub.value[:50])}'
                        )

            self.assertEqual(offenders, [], f'non-ASCII console output: {offenders}')
