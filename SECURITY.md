# PhishGuard AI — Security Design Document

## Summary
PhishGuard handles sensitive data: Gmail credentials and private email content.
This document explains every security measure implemented and why.

---

## 1. Authentication & Password Security

**Implementation:** Django's built-in auth system (`django.contrib.auth`)

**What it does:**
- Passwords are hashed using **PBKDF2 with SHA-256** — Django's default
- 260,000 iterations (as of Django 4.2) — computationally expensive to brute-force
- Passwords are **never stored in plaintext** anywhere in the codebase
- Salt is automatically added to every hash (prevents rainbow table attacks)

**Password strength rules enforced on registration:**
- Minimum 8 characters
- Cannot be too similar to username or email
- Cannot be a commonly used password (checked against a list of 20,000+)
- Cannot be entirely numeric

**Code location:** `phishguard/settings.py` → `AUTH_PASSWORD_VALIDATORS`

---

## 2. Gmail App Password — NOT the real Gmail password

**Why this matters:** Storing a user's actual Gmail password would be a critical security flaw.

**What we do instead:**
- Users generate a **16-character App Password** specifically for PhishGuard in their Google Account settings
- This token only grants IMAP read access — it cannot be used to log into Gmail, change account settings, or access Google Drive
- The token can be **revoked independently** of the main password at any time
- If PhishGuard's database were compromised, the attacker still cannot access the user's Gmail account fully

**Storage:** The App Password is stored **base64-encoded** in SQLite.

**Limitation acknowledged:** Base64 is encoding, not encryption. In a production deployment this would be replaced with:
- Django's `cryptography` library (Fernet symmetric encryption)
- The encryption key stored as an environment variable, never in code
- Example: `from cryptography.fernet import Fernet`

---

## 3. Brute-Force Login Protection

**Implementation:** `detector/middleware.py` → `LoginRateLimitMiddleware`

**What it does:**
- Intercepts every POST request to `/auth/login/`
- Tracks failed attempts per IP address
- After **5 failed attempts**, the IP is locked out for **5 minutes**
- Returns HTTP 429 (Too Many Requests) during lockout
- Successful login resets the counter

**Why it matters:** Without this, an attacker could try millions of password combinations automatically.

---

## 4. CSRF Protection (Cross-Site Request Forgery)

**Implementation:** Django's built-in `CsrfViewMiddleware`

**What it does:**
- Every HTML form includes a hidden `{% csrf_token %}` field
- Django validates this token on every POST request
- If the token is missing or wrong, the request is rejected (403 Forbidden)
- This prevents malicious websites from submitting forms on behalf of logged-in users

**All forms protected:** login, register, scan email, connect Gmail, profile edit, etc.

---

## 5. Session Security

**Settings applied:**
| Setting | Value | Reason |
|---------|-------|--------|
| `SESSION_COOKIE_HTTPONLY` | `True` | JavaScript cannot read the session cookie — prevents XSS session theft |
| `SESSION_COOKIE_AGE` | 28800 (8h) | Sessions expire after 8 hours of inactivity |
| `SESSION_COOKIE_SECURE` | `True` (production) | Cookie only sent over HTTPS |

---

## 6. Security Headers

Applied automatically by `django.middleware.security.SecurityMiddleware`:

| Header | Value | Protects Against |
|--------|-------|-----------------|
| `X-Content-Type-Options` | `nosniff` | MIME type sniffing attacks |
| `X-XSS-Protection` | `1; mode=block` | Reflected XSS in older browsers |
| `X-Frame-Options` | `DENY` | Clickjacking (site cannot be embedded in iframes) |
| `Strict-Transport-Security` | Set in production | Forces HTTPS |

---

## 7. Access Control — Users Only See Their Own Data

**Implementation:** All database queries are scoped to `request.user`

```python
# Every query filters by the logged-in user
emails = Email.objects.filter(user=request.user)

# Direct URL access to another user's email returns 404, not the data
email = get_object_or_404(Email, pk=pk, user=request.user)
```

This means even if a logged-in attacker guesses another user's email ID in the URL
(e.g. `/email/5/`), they get a 404 — not that user's email.

---

## 8. File Upload Validation

**Avatar uploads (profile pictures):**
- Only accepted MIME types: `image/jpeg`, `image/png`, `image/webp`, `image/gif`
- Maximum file size: **2 MB**
- Old avatar deleted from disk when replaced (prevents storage accumulation)

**Django upload limits in settings:**
- `DATA_UPLOAD_MAX_MEMORY_SIZE`: 5 MB
- `FILE_UPLOAD_MAX_MEMORY_SIZE`: 2 MB

---

## 9. SQL Injection Prevention

Django's ORM uses **parameterised queries** by default. Raw SQL is never used anywhere
in this codebase. All database interactions go through:
```python
Email.objects.filter(user=request.user, status='phishing')
Email.objects.create(sender=sender, ...)
```
Django escapes all values automatically — SQL injection is not possible through the ORM.

---

## 10. What Would Be Added in a Production Deployment

| Feature | Tool | Why |
|---------|------|-----|
| HTTPS everywhere | Let's Encrypt / Nginx | Encrypts all traffic in transit |
| Proper credential encryption | `cryptography.Fernet` | Encrypts App Passwords at rest |
| Rate limiting on all endpoints | `django-ratelimit` | Prevents API abuse |
| Audit logging | Django signals | Track who accessed what and when |
| Environment variables for secrets | `python-decouple` | No secrets in code |
| Database backups | cron + pg_dump | Data integrity |
| Penetration testing | OWASP ZAP | Find vulnerabilities before attackers |

---

## What to say to your supervisor

*"Security is addressed at multiple layers. At the credential level, we use Gmail App Passwords rather than real passwords, and Django's PBKDF2 password hashing for user accounts. At the application level, CSRF tokens protect all forms, session cookies are HttpOnly to prevent XSS theft, and a rate-limiting middleware blocks brute-force login attempts after 5 failures. At the data level, all database queries are user-scoped so users cannot access each other's data. For a production deployment we would add Fernet encryption for stored credentials, HTTPS enforcement, and environment-variable-based secret management."*
