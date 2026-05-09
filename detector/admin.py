from django.contrib import admin
from .models import Email, ScanLog, GmailAccount

@admin.register(GmailAccount)
class GmailAccountAdmin(admin.ModelAdmin):
    list_display = ('email_address', 'is_active', 'total_synced', 'last_synced', 'connected_at')
    list_filter = ('is_active',)

@admin.register(Email)
class EmailAdmin(admin.ModelAdmin):
    list_display = ('subject', 'sender', 'status', 'risk_percent', 'received_at', 'account')
    list_filter = ('status', 'account')
    search_fields = ('sender', 'subject', 'body')

@admin.register(ScanLog)
class ScanLogAdmin(admin.ModelAdmin):
    list_display = ('email', 'level', 'timestamp', 'message')
    list_filter = ('level',)
