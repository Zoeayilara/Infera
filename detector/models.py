"""
Infera AI — Database Models

Models:
  1. UserProfile   — extends Django's built-in User with avatar, bio, theme preference
  2. GmailAccount  — a connected Gmail inbox per user
  3. Email         — one scanned email + all ML results
  4. ScanLog       — log/alert feed
"""
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from django.db.models.signals import post_save
from django.dispatch import receiver
import base64
import os


class UserProfile(models.Model):
    """
    One-to-one extension of Django's built-in User model.
    Django handles username/password/email — we add the extra stuff.

    The post_save signal below auto-creates a profile whenever a new User is created,
    so you never have to create them manually.
    """
    THEME_CHOICES = [('light', 'Light'), ('dark', 'Dark')]

    user       = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    avatar     = models.ImageField(upload_to='avatars/', null=True, blank=True)
    bio        = models.CharField(max_length=200, blank=True)
    theme      = models.CharField(max_length=10, choices=THEME_CHOICES, default='light')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.user.username} profile'

    @property
    def avatar_url(self):
        """Returns avatar URL or a placeholder initial-based avatar."""
        if self.avatar and self.avatar.name:
            return self.avatar.url
        return None

    @property
    def initials(self):
        """e.g. 'Oluwaseun Adeyemi' → 'OA'"""
        parts = self.user.get_full_name().split()
        if len(parts) >= 2:
            return (parts[0][0] + parts[-1][0]).upper()
        return self.user.username[:2].upper()


# Auto-create a UserProfile whenever a new User is saved
@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)

@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    if hasattr(instance, 'profile'):
        instance.profile.save()


class GmailAccount(models.Model):
    """A Gmail inbox connected by a specific user."""
    user          = models.ForeignKey(User, on_delete=models.CASCADE,
                                      null=True, blank=True, related_name='gmail_accounts')
    email_address = models.EmailField()
    _app_password_encoded = models.TextField(db_column='app_password_encoded')
    connected_at  = models.DateTimeField(auto_now_add=True)
    last_synced   = models.DateTimeField(null=True, blank=True)
    is_active     = models.BooleanField(default=True)
    fetch_limit   = models.IntegerField(default=50)
    total_synced  = models.IntegerField(default=0)

    class Meta:
        ordering = ['-connected_at']
        unique_together = [('user', 'email_address')]

    def __str__(self):
        return self.email_address

    @property
    def app_password(self):
        try:
            return base64.b64decode(self._app_password_encoded).decode('utf-8')
        except Exception:
            return ''

    @app_password.setter
    def app_password(self, raw):
        self._app_password_encoded = base64.b64encode(
            raw.strip().replace(' ', '').encode()
        ).decode()


class Email(models.Model):
    STATUS_CHOICES = [
        ('phishing',   'Phishing'),
        ('suspicious', 'Suspicious'),
        ('safe',       'Safe'),
        ('pending',    'Pending'),
    ]

    account        = models.ForeignKey(GmailAccount, on_delete=models.CASCADE,
                                       null=True, blank=True, related_name='emails')
    user           = models.ForeignKey(User, on_delete=models.CASCADE,
                                       null=True, blank=True, related_name='emails')
    sender         = models.EmailField(max_length=255)
    sender_domain  = models.CharField(max_length=255, blank=True)
    subject        = models.CharField(max_length=500)
    body           = models.TextField()
    received_at    = models.DateTimeField(default=timezone.now)
    created_at     = models.DateTimeField(auto_now_add=True)
    message_id     = models.CharField(max_length=500, blank=True, db_index=True)

    status           = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    risk_score       = models.FloatField(default=0.0)
    text_score       = models.FloatField(default=0.0)
    url_score        = models.FloatField(default=0.0)
    metadata_score   = models.FloatField(default=0.0)
    attachment_score = models.FloatField(default=0.0)

    extracted_urls  = models.JSONField(default=list)
    has_attachment  = models.BooleanField(default=False)
    attachment_name = models.CharField(max_length=255, blank=True)
    why_flagged     = models.TextField(blank=True)
    is_read         = models.BooleanField(default=False)

    class Meta:
        ordering = ['-received_at']

    def __str__(self):
        return f'[{self.status.upper()}] {self.subject[:60]}'

    @property
    def risk_percent(self):      return round(self.risk_score * 100)
    @property
    def text_score_pct(self):    return round(self.text_score * 100)
    @property
    def url_score_pct(self):     return round(self.url_score * 100)
    @property
    def metadata_score_pct(self):return round(self.metadata_score * 100)
    @property
    def attachment_score_pct(self):return round(self.attachment_score * 100)


class ScanLog(models.Model):
    email     = models.ForeignKey(Email, on_delete=models.CASCADE, related_name='logs')
    timestamp = models.DateTimeField(auto_now_add=True)
    message   = models.TextField()
    level     = models.CharField(max_length=20, default='info')

    class Meta:
        ordering = ['-timestamp']
