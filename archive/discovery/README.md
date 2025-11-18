# Archived Discovery System

This directory contains the original complex discovery system that has been replaced by the simple Setup Wizard.

## What was here:

- `provider_discovery.py` - Complex orchestration with async/discovery modes
- `service_detector.py` - Async service detection with HTTP scanning
- `config_updater.py` - Complex configuration merging and management
- `__init__.py` - Module exports

## Why it was replaced:

The original system was over-engineered for the 90% use case:
- 1000+ lines of complex async code
- Multiple discovery modes (auto, manual, quick check)
- Complex interaction patterns and confirmation flows
- Difficult to maintain and debug

## New Simple System:

The new `SetupWizard` in `src/config/setup_wizard.py` provides:
- 200 lines of simple synchronous code
- 3-step interactive setup process
- Direct env file manipulation
- Focus on 80/20 rule - covers 90% of user needs

## When might you need this:

If you need the advanced features of the original system:
- Complex service detection endpoints
- Advanced configuration merging
- Multiple discovery modes
- Async processing capabilities

## Migration:

Most users should use the new `SetupWizard`. The archived system can be re-enabled if needed by:
1. Moving files back to `src/discovery/`
2. Updating imports in dependent files
3. Restoring any CLI integration

For details on the new system, see `src/config/setup_wizard.py`.