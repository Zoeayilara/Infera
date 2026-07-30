"""
Console output helpers for the management commands.

Named with a leading underscore so Django's command discovery skips it
(find_commands() ignores modules starting with '_').

The problem
-----------
Python on Windows uses UTF-8 for an *attached* console (PEP 528,
io._WindowsConsoleIO) but falls back to the locale encoding as soon as stdout
is piped or redirected, which is what happens under CI, a task scheduler, or
`manage.py ... > log.txt`. Writing a character the locale codepage cannot
encode then raises UnicodeEncodeError and kills the command mid-run.

Non-ASCII literals in our own output are fixed at the source, but email
subjects are interpolated into progress lines and those are attacker-supplied
text — there is no version of this codebase where they are guaranteed ASCII.

Why reconfigure rather than sanitising the data
-----------------------------------------------
Encoding the subject to ASCII would mangle it on every stream, including a
UTF-8 console that could have rendered it perfectly, and would only cover the
fields somebody remembered to wrap. Relaxing the stream's error handler instead
keeps full fidelity wherever the encoding allows it and degrades a single
unencodable character to '?' where it does not. The stored data is untouched;
only the rendering degrades, and it covers every interpolated field at once.

Verified on both backings: reconfigure(errors=...) is a TextIOWrapper method
and works whether the wrapper sits on FileIO (pipe/file) or on
_WindowsConsoleIO (real console device).
"""


def _relax(stream):
    """Set errors='replace' on the first stream in the chain that supports it."""
    for _ in range(4):
        if stream is None:
            return
        if hasattr(stream, 'reconfigure'):
            try:
                stream.reconfigure(errors='replace')
            except (ValueError, OSError):
                # Detached or already-closed stream: nothing to relax, and
                # failing to relax must never be what breaks the command.
                pass
            return
        # colorama, when it has wrapped sys.stdout, sits in front of the real
        # TextIOWrapper and has no reconfigure() of its own.
        stream = getattr(stream, 'wrapped', None)


def make_console_tolerant(*wrappers):
    """
    Let unencodable characters degrade instead of raising.

    Accepts Django OutputWrapper instances (self.stdout / self.stderr) or bare
    streams. A StringIO has neither reconfigure() nor .wrapped, so this is a
    no-op there and existing tests are unaffected.
    """
    for wrapper in wrappers:
        _relax(getattr(wrapper, '_out', wrapper))
