# Skills

This directory previously contained local skill files that duplicated functionality from the director plugin.

Those skills have been moved to `.archive/old-local-skills/` (2026-03-06).

## Active Skills

All director skills are now provided by the `director@claude-admin` plugin:

- `/director:dashboard` - Project dashboard generator
- `/director:director-resume` - Session orientation
- `/director:director-delegate` - Dispatch subagents
- `/director:director-task` - Launch background tasks
- `/director:choose`, `/director:plan`, `/director:pulse` - Project management
- And more - see full list in plugin

## Creating New Skills

To create project-specific skills that do not belong in the director plugin:
1. Create a markdown file in this directory
2. Follow the skill format (YAML frontmatter + markdown body)
3. Skills in this directory are auto-discovered by Claude Code
