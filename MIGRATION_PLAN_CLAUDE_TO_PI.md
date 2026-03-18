# Migration Plan: Conjecture — Claude Code → Pi Agent

**Created**: 2026-03-14
**Status**: DRAFT — awaiting review

---

## Overview

Migrate the Conjecture project from Claude Code (claude-admin director system) to pi agent. This involves replacing `.claude/` configuration with `.pi/`, migrating CLAUDE.md to PI.md, updating the container orchestration from claude-admin to pi-admin, and removing director-specific artifacts.

---

## Current State Inventory

### Claude Code Artifacts (to migrate or remove)

| Artifact | Location | Action |
|----------|----------|--------|
| `CLAUDE.md` | `/projects/conjecture/CLAUDE.md` | **Migrate** → `PI.md` |
| `.claude/settings.local.json` | `/projects/conjecture/.claude/` | **Migrate** → `.pi/agent/AGENTS.md` permissions |
| `.claude/commands/` | `/projects/conjecture/.claude/commands/` | **Remove** (empty) |
| `.claude/skills/` | `/projects/conjecture/.claude/skills/` | **Remove** (just a README pointing to director) |
| `.claude/worktrees/` | `/projects/conjecture/.claude/worktrees/` | **Remove** (empty) |
| `.director/` | `/projects/conjecture/.director/` | **Evaluate** — state.json, findings, plans |
| Director docker-compose | `claude-admin/instances/conjecture/` | **Replace** with pi-admin equivalent |
| Director inbox/outbox | `claude-admin/instances/conjecture/*.jsonl` | **Replace** with pi messaging |

### Files to Preserve (no changes)

| File | Reason |
|------|--------|
| `CHOICES.md` | Already pi-compatible (choose-wisely skill exists) |
| `PLAN.md` | Already pi-compatible (replan skill exists) |
| `MEMORY.md` | Session history — archive or adapt |
| `NEXT.md` | Backlog — keep as-is |
| `.conjecture/` | Project-specific config — independent of agent |

---

## Migration Phases

### Phase 1: Create `.pi/` Configuration

**Steps:**
1. Create `.pi/agent/AGENTS.md` with project guidelines extracted from `CLAUDE.md`
2. Map Claude permissions (`settings.local.json`) to pi equivalents in AGENTS.md
3. Register available skills (choose-wisely, replan, context7, debug-helper, web-search, web-fetch)
4. Run `/quick-setup` to validate pi detects the Python project correctly

**Content mapping for AGENTS.md:**
- Core philosophy (claims ≠ facts) → project guidelines
- Essential commands (pytest, benchmarks) → development commands section
- Architecture overview → architecture section
- Testing strategy → testing section
- Troubleshooting → troubleshooting section
- Statistical requirements → data validation rules

**Key decision:** CLAUDE.md is 400+ lines. AGENTS.md should be leaner — move reference docs (benchmark tables, troubleshooting, worktree patterns) to `.pi/docs/` or keep in existing `docs/`.

### Phase 2: Create `PI.md` (Project Instructions)

**Steps:**
1. Create `PI.md` at project root with concise agent instructions
2. Keep it focused: philosophy, commands, architecture, key patterns
3. Move verbose reference material (benchmark results, LM Studio workarounds, dataset issues) to `docs/`

### Phase 3: Migrate Director State

**Steps:**
1. Archive `.director/` contents to `.archive/director-state-backup/`
2. Extract any active state from `.director/state.json` and document in `MEMORY.md`
3. The director inbox/outbox/healthcheck pattern is replaced by pi's native messaging

**Decision needed:** Does the autonomous agent loop (state.json with IDLE/WAITING/BLOCKED) need a pi equivalent? If so, implement as a pi extension or custom tool.

### Phase 4: Update Container Orchestration

**Steps:**
1. Update `pi-admin/instances/conjecture/docker-compose.yml` to use pi container image
2. Remove Claude OAuth credential mounts
3. Remove cc-plugins/director volume mounts
4. Add pi-specific volume mounts (`.pi/` config, pi agent binary)
5. Update environment variables for pi

**Current claude-admin mounts to remove:**
- `~/.claude/.credentials.json` (Claude OAuth)
- `cc-plugins/` (director plugin code)
- `director.py`, `boot.py`, `healthMon.py`, etc.

**Note:** The pi-admin instance already exists at `/home/aaron/pi-admin/instances/conjecture/` with an empty `config.json`. This needs to be populated.

### Phase 5: Clean Up Claude Artifacts

**Steps:**
1. Remove `.claude/` directory
2. Remove or archive `CLAUDE.md`
3. Remove references to "Claude Code" in project docs (README.md, etc.)
4. Update any scripts referencing `claude` CLI commands
5. Git commit: `chore: migrate from claude code to pi agent`

### Phase 6: Validate

**Steps:**
1. Start pi agent against the project
2. Verify it reads PI.md and .pi/ config correctly
3. Run `/choose-wisely audit` to confirm skill works
4. Run test suite: `python -m pytest tests/ -m "unit" -v`
5. Verify pi can execute benchmarks and access all project tooling

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Director autonomous loop features lost | Medium | Document what director state machine did; build pi extension if needed |
| Inbox/outbox messaging broken | Medium | pi-admin already has inbox/outbox — verify compatibility |
| Claude-specific permissions not mapped | Low | Pi uses AGENTS.md guidelines, less restrictive by default |
| MEMORY.md format incompatible | Low | Pi doesn't enforce format — keep as-is |
| Active experiments interrupted | High | **Wait for running validations to complete before migration** |

---

## Open Questions for Review

1. **Director state machine**: The `.director/state.json` implements an autonomous agent loop (IDLE → WAITING → BLOCKED). Does pi need this, or is pi's native session model sufficient?

2. **Discord integration**: Director communicated via discord-pi-router. Is pi already wired to the same messaging bridge?

3. **Worktree parallelism**: Director used git worktrees for parallel experiments with subagents. Does pi support spawning parallel agent sessions?

4. **Background tasks**: Director had `run_in_background=true` with TaskOutput polling. What's the pi equivalent?

5. **Timing**: `.director/state.json` shows VALIDATING state with running experiments. Should we wait for those to complete, or migrate now and handle in-flight state?

6. **CLAUDE.md size**: The current CLAUDE.md is ~400 lines of dense project context. Should PI.md be equally detailed, or should we split into PI.md (essentials) + `.pi/docs/` (reference)?

---

## Recommended Execution Order

1. ✋ **Wait** for any in-flight director experiments to complete
2. **Phase 1** — Create `.pi/` config (non-destructive, additive)
3. **Phase 2** — Create `PI.md` (non-destructive, additive)
4. **Phase 6** — Validate pi works with new config (before removing anything)
5. **Phase 3** — Archive director state
6. **Phase 4** — Update container orchestration
7. **Phase 5** — Clean up Claude artifacts
8. Git tag: `v0.x-pi-migration-complete`
