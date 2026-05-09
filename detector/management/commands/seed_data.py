"""
Management command: python manage.py seed_data
Seeds the database with realistic phishing/safe email samples.
Based on patterns from: Nazario Phishing Corpus + Enron dataset
"""
from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import timedelta
import random

from detector.models import Email, ScanLog
from detector.ml_engine import classify_email

SAMPLE_EMAILS = [
    # ── PHISHING (Nazario-style) ──────────────────────────────────────────
    {
        "sender": "account-security@paypa1-support.net",
        "subject": "URGENT: Your PayPal account has been limited",
        "body": "Dear Customer,\n\nWe have detected suspicious activity on your PayPal account. Your account has been limited due to multiple failed login attempts.\n\nPlease click the link below to verify your identity within 24 hours or your account will be permanently suspended.\n\nVerify now: http://paypa1-support.net/verify?token=a8f3kx29\n\nFailure to act may result in permanent account closure.\n\n— PayPal Security Team",
        "attach": ""
    },
    {
        "sender": "security@hsbc-online-verify.com",
        "subject": "Your HSBC account requires immediate verification",
        "body": "Dear Account Holder,\n\nWe detected an unusual login on your HSBC account from Lagos, Nigeria. To secure your account, please verify your identity immediately.\n\nVerify account: http://hsbc-online-verify.com/secure-login\n\nFailure to verify within 48 hours will result in your account being locked.\n\n— HSBC Security",
        "attach": "account-statement.exe"
    },
    {
        "sender": "alerts@amaz0n-order-support.com",
        "subject": "Your Amazon order #304-1847291 has been cancelled",
        "body": "Dear Customer,\n\nWe were unable to process payment for your recent order. Your order has been cancelled.\n\nPlease update your payment information within 48 hours to avoid permanent account suspension.\n\nUpdate payment: http://amaz0n-order-support.com/payment-update\n\n— Amazon Customer Service",
        "attach": ""
    },
    {
        "sender": "admin@uniportal-stafflogin.xyz",
        "subject": "Staff portal login credentials update required",
        "body": "Dear Staff Member,\n\nOur IT department has upgraded the university staff portal. All users are required to re-enter their login credentials via the link below before Friday to maintain portal access.\n\nUpdate credentials: http://uniportal-stafflogin.xyz/staff-login\n\nPlease act now to avoid being locked out.\n\nIT Helpdesk",
        "attach": ""
    },
    {
        "sender": "billing@netflix-account-update.co",
        "subject": "Payment failed — update billing or lose access today",
        "body": "Dear Netflix Member,\n\nYour recent payment was declined. Please update your billing information immediately to continue your subscription.\n\nFailure to update within 24 hours will result in account suspension.\n\nUpdate billing: http://netflix-account-update.co/billing-update\n\n— Netflix Billing Team",
        "attach": ""
    },
    {
        "sender": "no-reply@facebookaccount-alert.tk",
        "subject": "Someone accessed your Facebook account from an unrecognised device",
        "body": "Hi,\n\nWe detected a login to your Facebook account from a new device in an unusual location.\n\nIf this was not you, please secure your account now by clicking below.\n\nSecure account: http://facebookaccount-alert.tk/recover-account\n\nIf you do not act within 2 hours, your account will be permanently disabled.\n\n— Facebook Security Team",
        "attach": ""
    },
    {
        "sender": "delivery@dhl-parcel-track.net",
        "subject": "Your DHL package could not be delivered — action required",
        "body": "Dear Customer,\n\nWe attempted to deliver your parcel but no one was available at the address.\n\nTo reschedule delivery, please click below and pay a N500 redelivery fee.\n\nReschedule: http://dhl-parcel-track.net/redeliver?id=88213\n\n— DHL Delivery Team",
        "attach": ""
    },
    {
        "sender": "support@apple-id-locked.com",
        "subject": "Your Apple ID has been locked — verify now",
        "body": "Dear Apple Customer,\n\nYour Apple ID was locked due to too many failed sign-in attempts. Unlock your account immediately to restore access to iCloud, App Store, and all Apple services.\n\nUnlock: http://apple-id-locked.com/verify?id=99821\n\n— Apple Support",
        "attach": ""
    },
    # ── SUSPICIOUS ────────────────────────────────────────────────────────
    {
        "sender": "internship@globalfund2026.org",
        "subject": "Congratulations — you have been selected for a USD 5,000 grant",
        "body": "Dear Applicant,\n\nAfter reviewing your profile, you have been selected for a USD 5,000 development grant from the Global Fund 2026.\n\nTo receive the funds, please complete the form at the link below and provide your bank account details for the transfer.\n\nClaim: http://globalfund2026.org/claim-grant",
        "attach": "grant-form.pdf"
    },
    {
        "sender": "hr@remote-jobs-worldwide.net",
        "subject": "Remote job offer — $3,000/month work from home",
        "body": "We found your profile online and would like to offer you a remote position at our company. Salary: $3,000/month, no experience required.\n\nTo proceed, please reply with your full name, address, and bank account details for payroll setup.\n\n— Global Recruitment Team",
        "attach": "offer_letter.pdf"
    },
    {
        "sender": "noreply@linkedln-jobs.com",
        "subject": "You have 3 new job matches — view now",
        "body": "Hi,\n\nYou have 3 new job recommendations matching your profile. Log in to view them.\n\nView jobs: http://linkedln-jobs.com/jobs\n\n— LinkedIn Jobs",
        "attach": ""
    },
    # ── SAFE (Enron-style) ────────────────────────────────────────────────
    {
        "sender": "john.okafor@gmail.com",
        "subject": "Meeting notes from yesterday's project sync",
        "body": "Hi team,\n\nPlease find attached the notes from our project meeting yesterday. Let me know if I missed anything important.\n\nBest regards,\nJohn",
        "attach": "meeting-notes.pdf"
    },
    {
        "sender": "newsletter@techcrunch.com",
        "subject": "This week in AI — top stories",
        "body": "Here are this week's top stories from TechCrunch:\n\n• Major AI funding rounds announced\n• New open-source model releases\n• Regulation updates from the EU\n\nRead more at https://techcrunch.com\n\nUnsubscribe from this newsletter",
        "attach": ""
    },
    {
        "sender": "registrar@uniabuja.edu.ng",
        "subject": "Second semester examination timetable 2026",
        "body": "Dear Students,\n\nPlease find the second semester examination timetable attached. For any queries, contact the registry office.\n\nKind regards,\nUniversity of Abuja Registrar's Office",
        "attach": "timetable_2026.pdf"
    },
    {
        "sender": "noreply@github.com",
        "subject": "[Infera/main] Pull request opened by collaborator",
        "body": "A pull request was opened in Infera/main by your collaborator.\n\nView the pull request: https://github.com/Infera/main/pull/12\n\n— GitHub",
        "attach": ""
    },
    {
        "sender": "admin@coursera.org",
        "subject": "Your certificate is ready — Deep Learning Specialization",
        "body": "Congratulations on completing Deep Learning Specialization!\n\nYour certificate is ready to download and share.\n\nhttps://coursera.org/certificates/abc123\n\nWe hope to see you in your next course.\n\n— Coursera Team",
        "attach": ""
    },
    {
        "sender": "amaka.ibrahim@gmail.com",
        "subject": "Follow-up from the conference last week",
        "body": "Hi,\n\nIt was great meeting you at the conference last week. As discussed, I am attaching the research paper we talked about.\n\nLooking forward to collaborating.\n\nBest regards,\nAmaka",
        "attach": "research-paper.pdf"
    },
]


class Command(BaseCommand):
    help = 'Seed the database with sample phishing and safe emails'

    def handle(self, *args, **options):
        if Email.objects.exists():
            self.stdout.write(self.style.WARNING(
                'Database already has data. Run with --flush to reset.'
            ))
            if '--flush' not in (options.get('args') or []):
                return

        self.stdout.write('Seeding sample emails...\n')
        base_time = timezone.now() - timedelta(days=7)
        counts = {'phishing': 0, 'suspicious': 0, 'safe': 0}

        for sample in SAMPLE_EMAILS:
            result = classify_email(
                sender=sample['sender'],
                subject=sample['subject'],
                body=sample['body'],
                attachment=sample.get('attach', ''),
            )
            offset = timedelta(
                hours=random.randint(0, 160),
                minutes=random.randint(0, 59)
            )
            email = Email.objects.create(
                sender=sample['sender'],
                sender_domain=result['sender_domain'],
                subject=sample['subject'],
                body=sample['body'],
                status=result['status'],
                risk_score=result['risk_score'],
                text_score=result['text_score'],
                url_score=result['url_score'],
                metadata_score=result['metadata_score'],
                attachment_score=result['attachment_score'],
                extracted_urls=result['extracted_urls'],
                has_attachment=bool(sample.get('attach')),
                attachment_name=sample.get('attach', ''),
                why_flagged=result['why_flagged'],
                received_at=base_time + offset,
            )
            ScanLog.objects.create(
                email=email,
                level='danger' if result['status'] == 'phishing'
                      else 'warning' if result['status'] == 'suspicious' else 'info',
                message=f"[{result['status'].upper()}] {sample['subject'][:65]} "
                        f"— score {round(result['risk_score'] * 100)}%",
                timestamp=base_time + offset,
            )
            counts[result['status']] = counts.get(result['status'], 0) + 1
            colour = self.style.ERROR if result['status'] == 'phishing' \
                     else self.style.WARNING if result['status'] == 'suspicious' \
                     else self.style.SUCCESS
            self.stdout.write(
                colour(f"  [{result['status'].upper():10}] {round(result['risk_score']*100):3}%  {sample['subject'][:55]}")
            )

        self.stdout.write(self.style.SUCCESS(
            f"\nDone! Seeded {len(SAMPLE_EMAILS)} emails: "
            f"{counts.get('phishing',0)} phishing, "
            f"{counts.get('suspicious',0)} suspicious, "
            f"{counts.get('safe',0)} safe."
        ))
