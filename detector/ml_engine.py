"""
Infera AI — Multi-Modal Detection Engine
=============================================
Uses a trained MLP Neural Network (deep learning) as the primary classifier,
with a rule-based fallback if the model hasn't been trained yet.

The neural network was trained on TF-IDF text embeddings + 12 hand-crafted
multi-modal features (URL, metadata, attachment signals).

To train the model: python ml/train_model.py
"""

import re
import os
import pickle
import numpy as np
from pathlib import Path

# ── Model loading ─────────────────────────────────────────────────────────────
# Paths to the saved trained model files
BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / 'ml' / 'phishguard_model.pkl'
VECT_PATH  = BASE_DIR / 'ml' / 'tfidf_vectorizer.pkl'

_model      = None
_vectorizer = None
_model_loaded = False


def _load_model():
    """Load the trained MLP model and TF-IDF vectorizer from disk."""
    global _model, _vectorizer, _model_loaded
    if _model_loaded:
        return _model is not None
    try:
        with open(MODEL_PATH, 'rb') as f:
            _model = pickle.load(f)
        with open(VECT_PATH, 'rb') as f:
            _vectorizer = pickle.load(f)
        _model_loaded = True
        return True
    except FileNotFoundError:
        _model_loaded = True   # don't try again
        return False
    except Exception:
        _model_loaded = True
        return False


# ── Feature extraction ────────────────────────────────────────────────────────

SUSPICIOUS_TLDS = {'.xyz', '.top', '.click', '.link', '.online', '.site',
                   '.tk', '.ml', '.ga', '.cf', '.gq', '.pw', '.cc', '.ws'}

BRAND_PATTERNS = [
    ('paypal',    ['paypa1', 'paypai', 'paypal-']),
    ('amazon',    ['amaz0n', 'arnazon', 'amazon-']),
    ('microsoft', ['micros0ft', 'microsoft-']),
    ('apple',     ['app1e', 'apple-', 'appleid-']),
    ('google',    ['g00gle', 'google-']),
    ('netflix',   ['netflix-', 'netflixbilling']),
    ('facebook',  ['faceb00k', 'facebook-']),
    ('linkedin',  ['linkedln', 'linked-in']),
    ('dhl',       ['dhl-']),
    ('hsbc',      ['hsbc-']),
    ('bank',      []),
]

MALICIOUS_EXTENSIONS = {'.exe', '.bat', '.cmd', '.scr', '.vbs',
                         '.js', '.jar', '.msi', '.ps1', '.hta'}

PHISH_KEYWORDS = [
    'verify your account', 'confirm your identity', 'update your payment',
    'account suspended', 'account limited', 'unusual activity',
    'within 24 hours', 'within 48 hours', 'failure to act',
    'permanently suspended', 'permanently closed', 'dear customer',
    'dear user', 'dear valued', 'click here to verify', 'verify now',
    'confirm now', 'act now', 'urgent action', 'bank details',
    'provide your bank', 'wire transfer', 'claim your prize',
]

SAFE_KEYWORDS = [
    'best regards', 'kind regards', 'meeting notes', 'attached please find',
    'as discussed', 'following up', 'newsletter', 'unsubscribe',
    'pull request', 'certificate is ready',
]


# ── Verdict bands — the single source of truth ────────────────────────────────
# risk_score is the number the detail page shows, and status is derived from it
# by status_from_risk() and nowhere else. Both classifier paths return a risk
# and let classify_email() label it, so the badge and the percentage cannot
# disagree. Anything that wants to change the verdict must move the risk.

RISK_PHISHING   = 0.55
RISK_SUSPICIOUS = 0.28


def status_from_risk(risk):
    """Map a risk score to a verdict label. The only place this mapping lives."""
    if risk >= RISK_PHISHING:
        return 'phishing'
    if risk >= RISK_SUSPICIOUS:
        return 'suspicious'
    return 'safe'


def _is_brand_impersonation(domain):
    d = domain.lower()
    TRUSTED = ['paypal.com', 'amazon.com', 'microsoft.com', 'apple.com',
                'google.com', 'netflix.com', 'facebook.com', 'linkedin.com',
                'dhl.com', 'hsbc.com', 'github.com', 'coursera.org']
    if any(d == s or d.endswith('.' + s) for s in TRUSTED):
        return False, None
    for brand, typos in BRAND_PATTERNS:
        if brand in d:
            return True, brand
        for t in typos:
            if t in d:
                return True, brand
    return False, None


# ── Known blind spots of the neural network ──────────────────────────────────
# The 12 features below are every structured signal the model receives, and all
# of them are derived from text. Two signals that this module computes and the
# UI displays never reach the model at all:
#
#   * Sender-domain impersonation / suspicious sender TLD.
#     _score_metadata() and _build_why_flagged() both detect it, so the detail
#     page can print "Sender domain impersonates dhl" on an email the model
#     scored 'safe'. This used to be fed in at index 10 — a slot the model had
#     been trained to read as a bulk-mail *safe* marker, which inverted it:
#     firing the impersonation signal pushed the verdict towards safe. It is
#     now correctly absent rather than actively harmful.
#
#   * Attachments. `attachment` is accepted below and deliberately unused —
#     training has no attachment feature to mirror. _score_attachment() returns
#     0.92 for a .exe and the page prints "Dangerous executable attachment",
#     but the model's output is unaffected.
#
# Closing either gap properly means adding a feature on BOTH sides and
# retraining. It cannot be done in this file alone, and doing it here alone is
# what caused the inversion above. See FeatureParityTests in detector/tests.py.
#
# In the meantime _escalate() floors the risk score on the attachment signal, so
# a .exe no longer lands as 'safe'. That is a patch over the blind spot, not a
# fix for it: the model still cannot weigh the attachment against anything else,
# and the sender blind spot is still uncovered — a domain impersonating a brand
# only moves the verdict when it arrives alongside an executable.


def _feature_text(subject, body, urls):
    """
    Rebuild the flat text blob the model was trained on.

    Training samples carry their URLs inline in the body
    ("...verify now http://paypa1-support.net/verify"), and every training
    feature is a regex over that one string. At serving time the URLs arrive
    as a separate list — recovered from HTML href attributes, where they never
    appear in the body text — so they have to be folded back in for the URL
    features to fire at all. Links already inline are not repeated.
    """
    text = subject + ' ' + body
    hidden_urls = [u for u in (urls or []) if u not in body]
    if hidden_urls:
        text += ' ' + ' '.join(hidden_urls)
    return text.lower()


def extract_hand_crafted_features(subject, body, sender, urls, attachment):
    """
    Extract the 12 numeric features for the neural network input.

    Every regex here is byte-for-byte the one in ml/train_model.py, at the same
    index, because the saved .pkl was fitted against those exact definitions —
    a feature that means something different here than it did during training
    is read by the network as the trained meaning.

    This is enforced by FeatureParityTests in detector/tests.py, which runs
    fixtures through both extractors and compares the vectors. Change a regex
    here without changing train_model.py (or vice versa) and that test fails.
    Do not rely on this docstring; an earlier one claimed parity that had
    silently stopped being true.

    `sender` and `attachment` are accepted but unused — see the blind-spot note
    above.
    """
    t = _feature_text(subject, body, urls)

    features = [
        # 0,1 — URL presence and count
        1 if re.search(r'https?://', t) else 0,
        min(len(re.findall(r'https?://', t)), 5) / 5,

        # 2,3 — Suspicious domain signals
        1 if re.search(r'http://[^\s]*\.(xyz|tk|ml|ga|cf|pw|top|click)', t) else 0,
        1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', t) else 0,

        # 4,5 — Urgency signals
        1 if re.search(r'\b(urgent|immediately|act now|within 24|within 48|expire)\b', t) else 0,
        1 if re.search(r'\b(suspended|limited|blocked|locked|compromised)\b', t) else 0,

        # 6,7 — Credential / financial request
        1 if re.search(r'\b(password|credential|login|verify|confirm|bank detail|account number)\b', t) else 0,
        1 if re.search(r'\b(click here|click below|click link|verify now|confirm now)\b', t) else 0,

        # 8 — Generic greeting (phishing signal)
        1 if re.search(r'\b(dear customer|dear user|dear valued|dear account holder)\b', t) else 0,

        # 9,10 — Safe signals. Index 10 is the bulk-mail marker, NOT a sender
        # signal — see the blind-spot note above before touching it.
        1 if re.search(r'\b(best regards|kind regards|attached|meeting|project|team)\b', t) else 0,
        1 if re.search(r'\b(unsubscribe|newsletter|view in browser)\b', t) else 0,

        # 11 — Text length normalised
        min(len(t.split()), 200) / 200,
    ]
    return features


def _neural_network_classify(sender, subject, body, urls, attachment):
    """
    Use the trained MLP neural network to score the email.
    Returns (risk_score, confidence_scores).

    Deliberately does not return a status. It used to label the email from
    argmax(probas) while risk_score was computed separately from the same
    probabilities, and the two could contradict each other: probabilities of
    (safe .49, suspicious .49, phishing .02) put argmax on 'suspicious' while
    the risk worked out to 26.5%, i.e. a Suspicious badge over a number inside
    the safe band. The caller now derives the label from the risk alone.
    """
    import scipy.sparse as sp

    # Build combined text for TF-IDF. Same URL-folding as the hand-crafted
    # features (see _feature_text), so a link that only ever existed inside an
    # HTML href still reaches the vectorizer as domain tokens. The sender is
    # appended for TF-IDF only — it is not part of the feature-vector text.
    combined_text = _feature_text(subject, body, urls) + ' ' + sender

    # TF-IDF features
    tfidf_vec = _vectorizer.transform([combined_text])

    # Hand-crafted multi-modal features
    hand_feat = np.array([extract_hand_crafted_features(
        subject, body, sender, urls, attachment
    )])

    # Combine into one feature vector (same as training)
    X = sp.hstack([tfidf_vec, sp.csr_matrix(hand_feat)])

    # Get probability scores from the neural network
    # probas shape: [1, 3] → [p_safe, p_suspicious, p_phishing]
    probas = _model.predict_proba(X)[0]

    p_safe, p_susp, p_phish = probas

    # The risk score is the phishing probability + half suspicious
    risk_score = round(float(p_phish + p_susp * 0.5), 3)

    return min(risk_score, 1.0), probas


# ── Rule-based fallback (used if model not trained yet) ───────────────────────

def _rule_based_classify(sender, subject, body, urls, attachment):
    """
    Fallback heuristic classifier used before model is trained.
    Returns (risk_score, flags). Like the network path, it does not label the
    email — classify_email() derives the status from the risk.
    """
    text = (subject + ' ' + body).lower()
    domain = sender.split('@')[-1].lower() if '@' in sender else sender.lower()

    score = 0.0
    flags = []

    # Text features
    phish_hits = sum(1 for kw in PHISH_KEYWORDS if kw in text)
    safe_hits  = sum(1 for kw in SAFE_KEYWORDS if kw in text)
    score += min(phish_hits * 0.14, 0.75)
    score -= min(safe_hits * 0.10, 0.35)

    if re.search(r'\b(urgent|immediately|act now|expire|suspended)\b', text):
        score += 0.12; flags.append('urgency language')
    if re.search(r'\b(click here|verify now|confirm now)\b', text):
        score += 0.12; flags.append('phishing CTA')
    if re.search(r'\b(password|credential|bank detail|pin\b)\b', text):
        score += 0.10; flags.append('credential request')

    # URL features
    for url in urls:
        tld = '.' + url.split('.')[-1].split('/')[0] if '.' in url else ''
        if tld in SUSPICIOUS_TLDS:
            score += 0.30; flags.append(f'suspicious TLD {tld}')
        is_imp, brand = _is_brand_impersonation(url.split('/')[0] if '/' in url else url)
        if is_imp:
            score += 0.40; flags.append(f'impersonates {brand}')

    # Sender metadata
    is_imp, brand = _is_brand_impersonation(domain)
    if is_imp:
        score += 0.45; flags.append(f'sender impersonates {brand}')
    sender_tld = '.' + domain.split('.')[-1] if '.' in domain else ''
    if sender_tld in SUSPICIOUS_TLDS:
        score += 0.35; flags.append(f'sender suspicious TLD')

    # Attachment
    if attachment:
        ext = os.path.splitext(attachment.lower())[1]
        if ext in MALICIOUS_EXTENSIONS:
            score += 0.90; flags.append(f'malicious attachment: {ext}')

    score = round(max(0.0, min(score, 1.0)), 3)
    return score, '; '.join(set(flags)) or 'No suspicious indicators'


# ── Per-modality scores (for the detail page breakdown) ──────────────────────

def _score_text(subject, body):
    text = (subject + ' ' + body).lower()
    score = 0.0
    hits = sum(1 for kw in PHISH_KEYWORDS if kw in text)
    score += min(hits * 0.14, 0.75)
    if re.search(r'\b(urgent|immediately|act now)\b', text): score += 0.12
    if re.search(r'\b(click here|verify now|confirm now)\b', text): score += 0.12
    if re.search(r'\b(password|credential|bank detail)\b', text): score += 0.10
    if re.search(r'\b(dear customer|dear user|dear valued)\b', text): score += 0.08
    safe = sum(1 for kw in SAFE_KEYWORDS if kw in text)
    score -= min(safe * 0.10, 0.35)
    return round(max(0.0, min(score, 1.0)), 3)


def _score_urls(urls):
    if not urls: return 0.0
    score = 0.0
    for url in urls:
        s = 0.0
        if url.startswith('http://'): s += 0.10
        tld = '.' + url.split('.')[-1].split('/')[0] if '.' in url else ''
        if tld in SUSPICIOUS_TLDS: s += 0.30
        is_imp, _ = _is_brand_impersonation(url.split('/')[0] if '://' not in url else url.split('://')[1].split('/')[0])
        if is_imp: s += 0.40
        if re.search(r'\d', url.split('/')[2] if len(url.split('/')) > 2 else ''): s += 0.15
        score = max(score, min(s, 1.0))
    return round(score, 3)


def _score_metadata(sender):
    domain = sender.split('@')[-1].lower() if '@' in sender else sender.lower()
    score = 0.0
    TRUSTED = ['gmail.com', 'yahoo.com', 'outlook.com', 'edu.ng', 'gov.ng',
                'microsoft.com', 'google.com', 'github.com', 'techcrunch.com']
    is_trusted = any(domain == t or domain.endswith('.' + t) for t in TRUSTED)
    tld = '.' + domain.split('.')[-1] if '.' in domain else ''
    if tld in SUSPICIOUS_TLDS: score += 0.35
    is_imp, _ = _is_brand_impersonation(domain)
    if is_imp: score += 0.45
    if re.search(r'\d', domain.split('.')[0]): score += 0.20
    if is_trusted: score = max(0.0, score - 0.25)
    return round(min(score, 1.0), 3)


def _score_attachment(filename):
    if not filename: return 0.0
    ext = os.path.splitext(filename.lower())[1]
    if ext in MALICIOUS_EXTENSIONS: return 0.92
    if ext in {'.zip', '.rar', '.7z'}: return 0.32
    if ext in {'.doc', '.docx', '.xls'}: return 0.20
    if ext == '.pdf':
        if re.search(r'(invoice|payment|urgent|verify)', filename.lower()): return 0.25
        return 0.05
    return 0.0


# ── Rule escalations for the model's blind spots ──────────────────────────────

def _dangerous_attachment(attachment):
    """Return the extension if it is an executable type, else None."""
    if not attachment:
        return None
    ext = os.path.splitext(attachment.lower())[1]
    return ext if ext in MALICIOUS_EXTENSIONS else None


def _escalate(risk, sender, attachment):
    """
    Raise the risk floor for signals the neural network provably cannot see.

    All 12 trained features are regexes over subject+body+URLs. There is no
    attachment feature on either side of train/serve, so a .exe cannot move the
    model's output at all — see the blind-spot note above. Until that is a real
    feature in ml/train_model.py and the model is retrained, the floor is
    applied here.

    Floors are the band minimums rather than fixed high numbers: the rule
    asserts "at least this bad" and never lowers a model score already above it.

    Returns (risk, [reason, ...]); the reasons are empty when nothing moved.
    """
    ext = _dangerous_attachment(attachment)
    if not ext:
        return risk, []

    domain = sender.split('@')[-1].lower() if '@' in sender else sender.lower()
    is_imp, brand = _is_brand_impersonation(domain)

    if is_imp:
        # An executable alone is dangerous but is not proof of phishing. An
        # executable from a domain impersonating a brand is.
        floor = RISK_PHISHING
        reason = (f'dangerous executable attachment ({ext}) from a sender '
                  f'domain impersonating {brand}')
    else:
        floor = RISK_SUSPICIOUS
        reason = f'dangerous executable attachment ({ext})'

    if risk >= floor:
        return risk, []
    return floor, [reason]


SOURCE_LABELS = {
    'neural_network': 'neural network',
    'rule_based':     'rule-based heuristics (model not trained)',
}


def _compose_why(signals, status, source, model_risk, final_risk, escalations):
    """
    Build the detail-page explanation, stating where the verdict came from.

    An escalated verdict and a model verdict are otherwise indistinguishable on
    the page, so the provenance is spelled out: which component decided, what
    the model said on its own, and which rule moved it.
    """
    src = SOURCE_LABELS.get(source, source)

    if escalations:
        provenance = (
            f'Verdict source: rule override — {"; ".join(escalations)}. '
            f'Risk raised to {round(final_risk * 100)}% '
            f'(the {src} alone scored this '
            f'{status_from_risk(model_risk)} at {round(model_risk * 100)}%).'
        )
    else:
        provenance = (f'Verdict source: {src} — '
                      f'{status} at {round(final_risk * 100)}% risk.')

    if signals:
        return provenance + '\nSignals: ' + '; '.join(signals)
    if status == 'safe':
        return (provenance +
                '\nNo suspicious indicators detected — email appears legitimate.')
    return provenance + '\nLow-confidence risk signals detected.'


def _detected_signals(sender, subject, body, urls, attachment):
    """
    Collect the human-readable signals present in the email.

    These are read off the same rule scorers that produce the modality scores,
    so they describe what the rules saw — not what the model weighed. Several
    of them (sender impersonation, attachments) are invisible to the model.
    """
    reasons = []
    text = (subject + ' ' + body).lower()
    domain = sender.split('@')[-1].lower() if '@' in sender else ''

    is_imp, brand = _is_brand_impersonation(domain)
    if is_imp:
        reasons.append(f'Sender domain impersonates {brand}')

    tld = '.' + domain.split('.')[-1] if '.' in domain else ''
    if tld in SUSPICIOUS_TLDS:
        reasons.append(f'Sender has suspicious domain extension ({tld})')

    if re.search(r'\b(urgent|immediately|act now|within 24|within 48)\b', text):
        reasons.append('Urgency/pressure language detected')

    if re.search(r'\b(click here|verify now|confirm now|update now)\b', text):
        reasons.append('Phishing call-to-action pattern')

    if re.search(r'\b(password|credential|bank detail|account number|pin\b)\b', text):
        reasons.append('Sensitive credential or financial information requested')

    if re.search(r'\b(dear customer|dear user|dear valued)\b', text):
        reasons.append('Generic impersonal greeting (not addressed by name)')

    for url in urls:
        url_domain = url.split('://')[-1].split('/')[0]
        is_url_imp, url_brand = _is_brand_impersonation(url_domain)
        if is_url_imp:
            reasons.append(f'URL impersonates {url_brand} ({url_domain})')
            break
        url_tld = '.' + url_domain.split('.')[-1] if '.' in url_domain else ''
        if url_tld in SUSPICIOUS_TLDS:
            reasons.append(f'Suspicious URL domain extension ({url_tld})')
            break

    if attachment:
        ext = os.path.splitext(attachment.lower())[1]
        if ext in MALICIOUS_EXTENSIONS:
            reasons.append(f'Dangerous executable attachment ({ext})')

    return reasons


# ── Main public API ───────────────────────────────────────────────────────────

def classify_email(sender: str, subject: str, body: str,
                   urls: list = None, attachment: str = '') -> dict:
    """
    Main classification function. Called by views.py for every email.

    1. Tries the trained MLP neural network first
    2. Falls back to rule-based heuristics if model not available
    3. Applies rule escalations for the model's blind spots
    4. Derives the status from the final risk score

    Step 4 is the only place a verdict label is produced, so `status` and
    `risk_score` cannot disagree — a caller that trusts the badge and a caller
    that trusts the percentage reach the same conclusion.

    Returns a dict with all scores and metadata.
    """
    if urls is None:
        urls = re.findall(r'https?://[^\s<>"\')+\]]+', body)

    # Compute per-modality scores (always, for the detail page)
    text_score       = _score_text(subject, body)
    url_score        = _score_urls(urls)
    metadata_score   = _score_metadata(sender)
    attachment_score = _score_attachment(attachment)

    sender_domain = sender.split('@')[-1].lower() if '@' in sender else sender.lower()

    # Try neural network first
    model_available = _load_model()

    if model_available:
        # Use the deep learning model
        model_risk, probas = _neural_network_classify(
            sender, subject, body, urls, attachment
        )
        source = 'neural_network'
    else:
        # Fallback to rule-based
        model_risk, _ = _rule_based_classify(
            sender, subject, body, urls, attachment
        )
        source = 'rule_based'

    # Floor the risk on blind-spot signals, then label the result. The status
    # is derived from the final risk and never set independently of it.
    risk_score, escalations = _escalate(model_risk, sender, attachment)
    risk_score = round(min(risk_score, 1.0), 3)
    status = status_from_risk(risk_score)

    signals = _detected_signals(sender, subject, body, urls, attachment)
    why = _compose_why(signals, status, source, model_risk, risk_score,
                       escalations)

    return {
        'status':           status,
        'risk_score':       risk_score,
        'model_risk_score': model_risk,
        'escalated':        bool(escalations),
        'text_score':       text_score,
        'url_score':        url_score,
        'metadata_score':   metadata_score,
        'attachment_score': attachment_score,
        'extracted_urls':   urls,
        'why_flagged':      why,
        'sender_domain':    sender_domain,
        'model_source':     source,
    }
