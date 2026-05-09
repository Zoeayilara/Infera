import json
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils import timezone
from django.db.models import Count, Q
from django.contrib import messages
from datetime import timedelta

from .models import Email, ScanLog, GmailAccount
from .ml_engine import classify_email
from .imap_connector import GmailIMAPConnector
from .sync_engine import sync_gmail_account


def _user_emails(request):
    """Return emails belonging to the logged-in user only."""
    return Email.objects.filter(user=request.user)


def dashboard(request):
    emails   = _user_emails(request)
    total    = emails.count()
    phishing = emails.filter(status='phishing').count()
    suspicious = emails.filter(status='suspicious').count()
    safe     = emails.filter(status='safe').count()

    recent_threats = emails.filter(
        status__in=['phishing', 'suspicious']
    ).order_by('-received_at')[:6]

    recent_alerts = ScanLog.objects.filter(
        email__user=request.user,
        level__in=['warning', 'danger']
    ).order_by('-timestamp')[:5]

    today = timezone.now().date()
    trend = []
    for i in range(6, -1, -1):
        day = today - timedelta(days=i)
        p = emails.filter(received_at__date=day, status='phishing').count()
        s = emails.filter(received_at__date=day, status='suspicious').count()
        trend.append({'day': day.strftime('%a'), 'phishing': p, 'suspicious': s})

    accounts = GmailAccount.objects.filter(user=request.user, is_active=True)

    context = {
        'total': total, 'phishing': phishing,
        'suspicious': suspicious, 'safe': safe,
        'recent_threats': recent_threats,
        'recent_alerts': recent_alerts,
        'trend_json': json.dumps(trend),
        'accounts': accounts,
    }
    return render(request, 'detector/dashboard.html', context)


def inbox(request):
    status_filter = request.GET.get('status', 'all')
    search = request.GET.get('q', '')
    emails = _user_emails(request)
    if status_filter != 'all':
        emails = emails.filter(status=status_filter)
    if search:
        emails = emails.filter(Q(sender__icontains=search) | Q(subject__icontains=search))
    return render(request, 'detector/inbox.html', {
        'emails': emails, 'status_filter': status_filter,
        'search': search, 'total_count': emails.count(),
    })


def email_detail(request, pk):
    email = get_object_or_404(Email, pk=pk, user=request.user)
    if not email.is_read:
        email.is_read = True
        email.save()
    return render(request, 'detector/email_detail.html', {'email': email})


def analytics(request):
    emails = _user_emails(request)
    total  = emails.count()
    phishing   = emails.filter(status='phishing').count()
    suspicious = emails.filter(status='suspicious').count()
    safe   = emails.filter(status='safe').count()

    top_senders = emails.filter(status='phishing').values(
        'sender_domain').annotate(count=Count('id')).order_by('-count')[:8]

    today = timezone.now().date()
    monthly = []
    for i in range(29, -1, -1):
        day = today - timedelta(days=i)
        monthly.append({'day': day.strftime('%b %d'),
                        'count': emails.filter(received_at__date=day, status='phishing').count()})

    return render(request, 'detector/analytics.html', {
        'total': total, 'phishing': phishing, 'suspicious': suspicious, 'safe': safe,
        'detection_rate': round(phishing / total * 100, 1) if total else 0,
        'top_senders': top_senders,
        'monthly_json': json.dumps(monthly),
    })


def settings_view(request):
    accounts = GmailAccount.objects.filter(user=request.user)
    return render(request, 'detector/settings.html', {'accounts': accounts})


def scan_email(request):
    if request.method == 'POST':
        sender     = request.POST.get('sender', '').strip()
        subject    = request.POST.get('subject', '').strip()
        body       = request.POST.get('body', '').strip()
        attachment = request.POST.get('attachment_name', '').strip()
        if not sender or not subject or not body:
            return render(request, 'detector/scan.html',
                          {'error': 'Sender, subject and body are all required.'})
        result = classify_email(sender, subject, body, attachment=attachment)
        email_obj = Email.objects.create(
            user=request.user, sender=sender,
            sender_domain=result['sender_domain'], subject=subject, body=body,
            status=result['status'], risk_score=result['risk_score'],
            text_score=result['text_score'], url_score=result['url_score'],
            metadata_score=result['metadata_score'],
            attachment_score=result['attachment_score'],
            extracted_urls=result['extracted_urls'],
            has_attachment=bool(attachment), attachment_name=attachment,
            why_flagged=result['why_flagged'],
        )
        ScanLog.objects.create(
            email=email_obj,
            level='danger' if result['status'] == 'phishing'
                  else 'warning' if result['status'] == 'suspicious' else 'info',
            message=f"[{result['status'].upper()}] {subject[:60]} — score {round(result['risk_score']*100)}%",
        )
        return redirect('email_detail', pk=email_obj.pk)
    return render(request, 'detector/scan.html')


def gmail_connect(request):
    if request.method == 'POST':
        email_address = request.POST.get('email_address', '').strip()
        app_password  = request.POST.get('app_password', '').strip()
        fetch_limit   = int(request.POST.get('fetch_limit', 50))
        if not email_address or not app_password:
            return render(request, 'detector/gmail_connect.html',
                          {'error': 'Both fields are required.'})
        connector = GmailIMAPConnector(email_address, app_password)
        ok, msg = connector.test_connection()
        if not ok:
            return render(request, 'detector/gmail_connect.html',
                          {'error': f'Could not connect: {msg}', 'email_address': email_address})
        account, created = GmailAccount.objects.get_or_create(
            user=request.user, email_address=email_address,
            defaults={'fetch_limit': fetch_limit}
        )
        account.app_password = app_password
        account.fetch_limit  = fetch_limit
        account.is_active    = True
        account.save()
        messages.success(request, f'{email_address} connected!')
        return redirect('gmail_sync', pk=account.pk)
    return render(request, 'detector/gmail_connect.html')


def gmail_sync(request, pk):
    account = get_object_or_404(GmailAccount, pk=pk, user=request.user)
    summary = sync_gmail_account(account, user=request.user)
    if summary['success']:
        # Build per-folder breakdown string
        folder_parts = []
        for folder_name, fdata in summary.get('folders', {}).items():
            if fdata.get('fetched', 0) > 0:
                folder_parts.append(
                    f"{folder_name}: {fdata['new']} new "
                    f"({fdata.get('phishing', 0)} phishing)"
                )
        folder_str = ' | '.join(folder_parts) if folder_parts else ''

        msg = (
            f"Sync complete — {summary['new']} new emails scanned: "
            f"{summary['phishing']} phishing, {summary['suspicious']} suspicious, "
            f"{summary['safe']} safe. {summary['skipped']} already seen."
        )
        if folder_str:
            msg += f" [{folder_str}]"
        messages.success(request, msg)

        if summary['phishing'] > 0:
            messages.warning(request,
                f"⚠️ {summary['phishing']} phishing email(s) detected — "
                f"check your inbox and spam folder results below.")
    else:
        messages.error(request,
            f"Sync failed: {summary['errors'][0] if summary['errors'] else 'Unknown error'}")
    return redirect('inbox')


def gmail_disconnect(request, pk):
    account = get_object_or_404(GmailAccount, pk=pk, user=request.user)
    addr = account.email_address
    account.delete()
    messages.success(request, f'{addr} disconnected.')
    return redirect('settings')


def gmail_accounts(request):
    accounts = GmailAccount.objects.filter(user=request.user)
    return render(request, 'detector/gmail_accounts.html', {'accounts': accounts})


def api_stats(request):
    qs = Email.objects.filter(user=request.user) if request.user.is_authenticated else Email.objects.none()
    return JsonResponse({
        'total':      qs.count(),
        'phishing':   qs.filter(status='phishing').count(),
        'suspicious': qs.filter(status='suspicious').count(),
        'safe':       qs.filter(status='safe').count(),
    })


@csrf_exempt
@require_http_methods(['POST'])
def api_scan(request):
    try:
        data = json.loads(request.body)
        result = classify_email(
            sender=data.get('sender', ''), subject=data.get('subject', ''),
            body=data.get('body', ''), attachment=data.get('attachment', ''),
        )
        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)


def api_recent_alerts(request):
    qs = ScanLog.objects.filter(email__user=request.user) if request.user.is_authenticated \
         else ScanLog.objects.none()
    alerts = qs.filter(level__in=['warning', 'danger']).order_by('-timestamp')[:10]
    return JsonResponse({'alerts': [
        {'message': a.message, 'level': a.level, 'time': a.timestamp.strftime('%H:%M')}
        for a in alerts
    ]})


def api_notifications(request):
    """
    Returns unread notification data for the bell icon.
    Includes recent phishing detections, sync results, anything alert-worthy.
    Marks them as 'seen' by storing count in session so the dot disappears.
    """
    if not request.user.is_authenticated:
        return JsonResponse({'count': 0, 'notifications': []})

    # Get recent danger/warning logs for this user
    recent = ScanLog.objects.filter(
        email__user=request.user,
        level__in=['danger', 'warning']
    ).order_by('-timestamp')[:10]

    # Count how many are new since last check
    last_seen_count = request.session.get('notif_last_seen', 0)
    total_threat_count = ScanLog.objects.filter(
        email__user=request.user,
        level__in=['danger', 'warning']
    ).count()
    new_count = max(0, total_threat_count - last_seen_count)

    notifs = []
    for log in recent:
        notifs.append({
            'id':      log.pk,
            'message': log.message,
            'level':   log.level,
            'time':    log.timestamp.strftime('%b %d, %H:%M'),
            'email_id': log.email_id,
        })

    return JsonResponse({
        'count':         new_count,
        'total':         total_threat_count,
        'notifications': notifs,
    })


def api_mark_notifications_seen(request):
    """Called when user opens the notification panel — clears the badge."""
    if request.method == 'POST' and request.user.is_authenticated:
        total = ScanLog.objects.filter(
            email__user=request.user,
            level__in=['danger', 'warning']
        ).count()
        request.session['notif_last_seen'] = total
        return JsonResponse({'ok': True})
    return JsonResponse({'ok': False})
