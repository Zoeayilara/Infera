"""
Infera AI — Authentication Views
Login, Register, Logout, Profile edit, Avatar upload.

We use Django's built-in auth system (User model, authenticate, login, logout).
We only need to write the views — Django handles password hashing, sessions etc.
"""
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.views.decorators.http import require_http_methods


def landing(request):
    """
    The public homepage — shown to visitors who are NOT logged in.
    If a logged-in user visits '/', they go straight to the dashboard.
    We handle that in urls.py by checking request.user.is_authenticated.
    """
    if request.user.is_authenticated:
        return redirect('dashboard')
    return render(request, 'detector/landing.html')


@require_http_methods(['GET', 'POST'])
def register_view(request):
    """
    Registration page.
    GET  → show blank form
    POST → validate, create User, log them in, redirect to dashboard
    """
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        first_name = request.POST.get('first_name', '').strip()
        last_name  = request.POST.get('last_name', '').strip()
        username   = request.POST.get('username', '').strip()
        email      = request.POST.get('email', '').strip()
        password1  = request.POST.get('password1', '')
        password2  = request.POST.get('password2', '')

        errors = []

        # Validate
        if not all([first_name, username, email, password1, password2]):
            errors.append('All fields except last name are required.')
        if password1 != password2:
            errors.append('Passwords do not match.')
        if len(password1) < 8:
            errors.append('Password must be at least 8 characters.')
        if User.objects.filter(username=username).exists():
            errors.append(f'Username "{username}" is already taken.')
        if User.objects.filter(email=email).exists():
            errors.append('An account with that email already exists.')

        if errors:
            return render(request, 'detector/auth/register.html', {
                'errors': errors,
                'first_name': first_name,
                'last_name': last_name,
                'username': username,
                'email': email,
            })

        # Create the user — Django hashes the password automatically
        user = User.objects.create_user(
            username=username,
            email=email,
            password=password1,
            first_name=first_name,
            last_name=last_name,
        )

        # Log them in straight away
        login(request, user)
        messages.success(request, f'Welcome to Infera, {first_name}!')
        return redirect('dashboard')

    return render(request, 'detector/auth/register.html')


@require_http_methods(['GET', 'POST'])
def login_view(request):
    """
    Login page.
    GET  → show blank form
    POST → authenticate, log in, redirect
    """
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')
        # 'next' is the page the user tried to visit before being redirected to login
        next_url = request.POST.get('next', '/')

        # Django's authenticate checks username+password against the database
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect(next_url if next_url.startswith('/') else '/')
        else:
            return render(request, 'detector/auth/login.html', {
                'error': 'Incorrect username or password.',
                'username': username,
                'next': next_url,
            })

    return render(request, 'detector/auth/login.html', {
        'next': request.GET.get('next', '/'),
    })


def logout_view(request):
    """Logs out and redirects to login page."""
    logout(request)
    messages.success(request, 'You have been logged out.')
    return redirect('login')


@login_required
def profile_view(request):
    """
    Profile page — shows user info and lets them edit it.
    GET  → show profile
    POST → save changes
    """
    profile = request.user.profile

    if request.method == 'POST':
        action = request.POST.get('action', 'edit')

        if action == 'edit':
            # Update User fields
            request.user.first_name = request.POST.get('first_name', '').strip()
            request.user.last_name  = request.POST.get('last_name', '').strip()
            request.user.email      = request.POST.get('email', '').strip()
            request.user.save()

            # Update Profile fields
            profile.bio = request.POST.get('bio', '').strip()[:200]
            profile.save()

            messages.success(request, 'Profile updated successfully.')

        elif action == 'avatar':
            # Handle avatar upload
            avatar_file = request.FILES.get('avatar')
            if avatar_file:
                # Validate file type
                allowed = ['image/jpeg', 'image/png', 'image/webp', 'image/gif']
                if avatar_file.content_type not in allowed:
                    messages.error(request, 'Only JPG, PNG, WebP or GIF images are allowed.')
                elif avatar_file.size > 2 * 1024 * 1024:
                    messages.error(request, 'Avatar must be under 2MB.')
                else:
                    # Delete old avatar to save space
                    if profile.avatar:
                        try:
                            import os
                            if os.path.exists(profile.avatar.path):
                                os.remove(profile.avatar.path)
                        except Exception:
                            pass
                    profile.avatar = avatar_file
                    profile.save()
                    messages.success(request, 'Profile picture updated.')

        elif action == 'password':
            old_pw  = request.POST.get('old_password', '')
            new_pw1 = request.POST.get('new_password1', '')
            new_pw2 = request.POST.get('new_password2', '')

            if not request.user.check_password(old_pw):
                messages.error(request, 'Current password is incorrect.')
            elif new_pw1 != new_pw2:
                messages.error(request, 'New passwords do not match.')
            elif len(new_pw1) < 8:
                messages.error(request, 'New password must be at least 8 characters.')
            else:
                request.user.set_password(new_pw1)
                request.user.save()
                # Re-authenticate so they stay logged in after password change
                from django.contrib.auth import update_session_auth_hash
                update_session_auth_hash(request, request.user)
                messages.success(request, 'Password changed successfully.')

        return redirect('profile')

    # Count their emails for profile stats
    email_count    = request.user.emails.count()
    phishing_count = request.user.emails.filter(status='phishing').count()
    gmail_count    = request.user.gmail_accounts.filter(is_active=True).count()

    return render(request, 'detector/auth/profile.html', {
        'profile': profile,
        'email_count': email_count,
        'phishing_count': phishing_count,
        'gmail_count': gmail_count,
    })


@login_required
def toggle_theme(request):
    """
    AJAX endpoint — called when user clicks the light/dark toggle.
    Saves preference to UserProfile so it persists across sessions.
    """
    if request.method == 'POST':
        profile = request.user.profile
        profile.theme = 'dark' if profile.theme == 'light' else 'light'
        profile.save()
        from django.http import JsonResponse
        return JsonResponse({'theme': profile.theme})
    return redirect('dashboard')
