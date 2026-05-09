# PhishGuard AI — Final Year Project
## Multi-Modal Deep Learning for Email Phishing Detection
### With Live Gmail IMAP Integration

---

## Project Structure
```
phishguard/
├── manage.py
├── requirements.txt
├── README.md
├── dataset/sample_emails.csv
├── phishguard/
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
└── detector/
    ├── models.py           ← GmailAccount + Email + ScanLog
    ├── views.py            ← All pages + Gmail views + API
    ├── ml_engine.py        ← Multi-modal detection (text+URL+meta+attach)
    ├── imap_connector.py   ← Connects to Gmail via IMAP SSL
    ├── sync_engine.py      ← Orchestrates fetch → classify → save pipeline
    ├── urls.py
    ├── admin.py
    ├── management/commands/seed_data.py
    └── templates/detector/
        ├── base.html
        ├── dashboard.html
        ├── inbox.html
        ├── email_detail.html
        ├── scan.html
        ├── analytics.html
        ├── settings.html
        ├── gmail_connect.html   ← NEW: Gmail connection form + instructions
        └── gmail_accounts.html  ← NEW: Manage connected accounts
```

---

## Quick Setup

```bash
pip install Django
python manage.py makemigrations
python manage.py migrate
python manage.py seed_data        # optional: loads 17 sample emails
python manage.py createsuperuser
python manage.py runserver
```

Open: http://127.0.0.1:8000

---

## How to Connect Gmail (the user does this once)

1. Go to myaccount.google.com/security
2. Turn ON 2-Step Verification
3. Go to myaccount.google.com/apppasswords
4. Type "PhishGuard" → click Create
5. Copy the 16-character password shown
6. In PhishGuard: go to Settings → Connect Gmail → paste it

PhishGuard then connects to imap.gmail.com:993, fetches your inbox,
and scans each email through the multi-modal AI engine automatically.

---

## Pages

| URL | Description |
|-----|-------------|
| `/` | Dashboard — stats, 7-day trend, recent alerts |
| `/inbox/` | All scanned emails, filterable |
| `/email/<id>/` | Full AI breakdown per email |
| `/scan/` | Manual scan (paste email content) |
| `/analytics/` | Charts, model metrics |
| `/gmail/connect/` | Connect a Gmail account |
| `/gmail/accounts/` | Manage connected accounts |
| `/gmail/sync/<id>/` | Trigger a sync |
| `/api/scan/` | REST API (POST JSON) |

---

## Multi-Modal Detection Engine

Four modalities fused with weighted scoring:

| Modality | Weight | What it checks |
|----------|--------|----------------|
| Text embedding | 40% | Keywords, urgency, credential requests |
| URL features | 30% | Domain spoofing, TLD, brand impersonation |
| Sender metadata | 20% | Domain reputation, typosquatting |
| Attachment scan | 10% | File extension risk, filename keywords |

Thresholds: ≥55% = Phishing | 28–54% = Suspicious | <28% = Safe

---

## Dataset Reference
- Phishing patterns: Nazario Phishing Email Corpus
- Legitimate patterns: Enron email dataset  
- Curated dataset: https://figshare.com/articles/dataset/Curated_Dataset_-_Phishing_Email/24899952
