"""
Infera AI — Security Middleware
=====================================
LoginRateLimitMiddleware: brute-force login protection.
After 5 failed attempts from the same IP, lock out for 5 minutes.
"""
import time
from django.http import HttpResponse
from django.conf import settings

_login_attempts = {}  # { ip: {count, first_fail, locked_until} }


class LoginRateLimitMiddleware:

    def __init__(self, get_response):
        self.get_response  = get_response
        self.max_attempts  = getattr(settings, 'LOGIN_MAX_ATTEMPTS', 5)
        self.lockout_secs  = getattr(settings, 'LOGIN_LOCKOUT_SECS', 300)
        self.login_url     = '/auth/login/'

    def __call__(self, request):
        if request.method == 'POST' and request.path == self.login_url:
            ip  = self._get_ip(request)
            now = time.time()
            entry = _login_attempts.get(ip, {})

            locked_until = entry.get('locked_until', 0)
            if now < locked_until:
                remaining = int(locked_until - now)
                return HttpResponse(
                    f'Too many failed login attempts. Wait {remaining} seconds.',
                    status=429, content_type='text/plain',
                )

            response = self.get_response(request)

            if response.status_code == 200:
                # Still on login page = failed attempt
                count = entry.get('count', 0) + 1
                _login_attempts[ip] = {
                    'count':        count,
                    'first_fail':   entry.get('first_fail', now),
                    'locked_until': now + self.lockout_secs if count >= self.max_attempts else 0,
                }
            elif response.status_code in (301, 302):
                # Redirect = success — clear counter
                _login_attempts.pop(ip, None)

            return response

        return self.get_response(request)

    def _get_ip(self, request):
        xff = request.META.get('HTTP_X_FORWARDED_FOR')
        return xff.split(',')[0].strip() if xff else request.META.get('REMOTE_ADDR', '0.0.0.0')
