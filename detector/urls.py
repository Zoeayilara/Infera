from django.urls import path
from django.contrib.auth.decorators import login_required
from . import views, auth_views

urlpatterns = [
    # Public
    path('', auth_views.landing, name='landing'),

    # Auth
    path('auth/register/', auth_views.register_view, name='register'),
    path('auth/login/',    auth_views.login_view,    name='login'),
    path('auth/logout/',   auth_views.logout_view,   name='logout'),
    path('auth/profile/',  auth_views.profile_view,  name='profile'),
    path('auth/theme/',    auth_views.toggle_theme,  name='toggle_theme'),

    # App (login required)
    path('dashboard/', login_required(views.dashboard),    name='dashboard'),
    path('inbox/',     login_required(views.inbox),        name='inbox'),
    path('email/<int:pk>/', login_required(views.email_detail), name='email_detail'),
    path('analytics/', login_required(views.analytics),   name='analytics'),
    path('settings/',  login_required(views.settings_view), name='settings'),
    path('scan/',      login_required(views.scan_email),   name='scan_email'),

    # Gmail IMAP
    path('gmail/connect/',             login_required(views.gmail_connect),    name='gmail_connect'),
    path('gmail/accounts/',            login_required(views.gmail_accounts),   name='gmail_accounts'),
    path('gmail/sync/<int:pk>/',       login_required(views.gmail_sync),       name='gmail_sync'),
    path('gmail/disconnect/<int:pk>/', login_required(views.gmail_disconnect), name='gmail_disconnect'),

    # API
    path('api/stats/',                  views.api_stats,                   name='api_stats'),
    path('api/scan/',                   views.api_scan,                    name='api_scan'),
    path('api/recent-alerts/',          views.api_recent_alerts,           name='api_recent_alerts'),
    path('api/notifications/',          views.api_notifications,           name='api_notifications'),
    path('api/notifications/seen/',     views.api_mark_notifications_seen, name='api_notif_seen'),
]
