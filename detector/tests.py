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
import re

from django.test import SimpleTestCase

from ml.train_model import extract_hand_crafted_features as training_features

from .imap_connector import (
    extract_body_and_attachments,
    extract_urls_from_html,
    extract_urls_from_text,
)
from .ml_engine import (
    _feature_text,
    classify_email,
    extract_hand_crafted_features as serving_features,
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
