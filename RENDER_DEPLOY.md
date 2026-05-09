# Deploying Infera to Render

## Quick Deploy

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Render will auto-detect the `render.yaml` config

## Manual Setup (if not using render.yaml)

- **Runtime**: Python
- **Build Command**: `pip install -r requirements.txt && python manage.py collectstatic --noinput && python manage.py migrate`
- **Start Command**: `gunicorn infera_config.wsgi:application --bind 0.0.0.0:$PORT --workers 2`

## Environment Variables (set in Render dashboard)

| Variable | Value |
|----------|-------|
| `DJANGO_SECRET_KEY` | Generate a strong random key |
| `RENDER` | `true` |
| `GOOGLE_CLIENT_ID` | Your Google OAuth client ID (for Gmail) |
| `GOOGLE_CLIENT_SECRET` | Your Google OAuth secret (for Gmail) |

## After Deploy

- Your app will be live at `https://infera.onrender.com` (or your chosen name)
- Run `python manage.py createsuperuser` via Render Shell to create admin
