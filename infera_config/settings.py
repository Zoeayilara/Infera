from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY: In production load from environment variable
# SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY')
SECRET_KEY = 'django-insecure-phishguard-fyp-2026-change-in-production'

DEBUG = True
ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'detector',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',        # security headers
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',            # CSRF on all POSTs
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',  # anti-clickjacking
    'detector.middleware.LoginRateLimitMiddleware',         # brute-force protection
]

ROOT_URLCONF = 'infera_config.urls'

TEMPLATES = [{
    'BACKEND': 'django.template.backends.django.DjangoTemplates',
    'DIRS': [], 'APP_DIRS': True,
    'OPTIONS': {'context_processors': [
        'django.template.context_processors.debug',
        'django.template.context_processors.request',
        'django.contrib.auth.context_processors.auth',
        'django.contrib.messages.context_processors.messages',
    ]},
}]

WSGI_APPLICATION = 'infera_config.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Password strength rules enforced on registration + password change
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
     'OPTIONS': {'min_length': 8}},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# Session security
SESSION_COOKIE_HTTPONLY = True      # JS cannot read session cookie (XSS protection)
SESSION_COOKIE_AGE      = 28800     # 8 hours
# In production with HTTPS: SESSION_COOKIE_SECURE = True

# CSRF (must be False so AJAX can read the token)
CSRF_COOKIE_HTTPONLY = False

# Security headers (applied by SecurityMiddleware)
SECURE_CONTENT_TYPE_NOSNIFF = True  # no MIME sniffing
SECURE_BROWSER_XSS_FILTER   = True  # browser XSS filter
X_FRAME_OPTIONS             = 'DENY'  # no embedding in iframes
# In production with HTTPS:
# SECURE_SSL_REDIRECT = True
# SECURE_HSTS_SECONDS = 31536000

# Login brute-force limits
LOGIN_MAX_ATTEMPTS = 5
LOGIN_LOCKOUT_SECS = 300   # 5 minutes

DATA_UPLOAD_MAX_MEMORY_SIZE = 5242880   # 5 MB
FILE_UPLOAD_MAX_MEMORY_SIZE = 2097152   # 2 MB

LANGUAGE_CODE = 'en-us'
TIME_ZONE     = 'Africa/Lagos'
USE_I18N = True
USE_TZ   = True

STATIC_URL = '/static/'
MEDIA_URL  = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

LOGIN_URL           = '/auth/login/'
LOGIN_REDIRECT_URL  = '/dashboard/'
LOGOUT_REDIRECT_URL = '/auth/login/'

# ── Render / Production settings ─────────────────────────────────────────
import os as _os

if _os.environ.get('RENDER'):
    DEBUG = False
    SECRET_KEY = _os.environ.get('DJANGO_SECRET_KEY', SECRET_KEY)
    ALLOWED_HOSTS = [_os.environ.get('RENDER_EXTERNAL_HOSTNAME', '*')]
    
    # Static files for production
    STATIC_ROOT = BASE_DIR / 'staticfiles'
    STATICFILES_STORAGE = 'django.contrib.staticfiles.storage.ManifestStaticFilesStorage'
    
    # Security
    SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
    SECURE_SSL_REDIRECT = True
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
