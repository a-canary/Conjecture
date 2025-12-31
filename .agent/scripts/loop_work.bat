@echo off
REM Loop-work execution wrapper with UTF-8 encoding fix
REM Created 2025-12-29 to resolve Unicode encoding errors in agent outputs
REM Issue: Windows charmap encoding blocks Unicode character writes (→ arrow, ✓ checkmark)
REM Fix: Set PYTHONIOENCODING=utf-8 before opencode invocation

echo Setting UTF-8 encoding environment variable...
set PYTHONIOENCODING=utf-8

echo Verifying encoding setting...
echo PYTHONIOENCODING=%PYTHONIOENCODING%

echo Starting opencode loop-work with UTF-8 encoding...
opencode run loop-work --agent ego

echo Loop-work completed with encoding: %PYTHONIOENCODING%