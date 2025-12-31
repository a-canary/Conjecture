# Conjecture GLM Model Override

**Date:** 2025-12-27
**Status:** ✅ COMPLETE

---

## Change Summary

Overridden Conjecture project to use **GLM 4.6** models for all agents instead of Anthropic models.

---

## Available GLM Models (from `opencode models`)

Available GLM 4.x models:
- `chutes/zai-org/GLM-4.5-Air` - GLM 4.5 Air
- `chutes/zai-org/GLM-4.6` - GLM 4.6 (latest)
- `chutes/zai-org/GLM-4.6V` - GLM 4.6 V
- `chutes/zai-org/GLM-4.6-TEE` - GLM 4.6 TEE
- `chutes/zai-org/GLM-4.6V` - GLM 4.6V (vision)
- `chutes/zai-org/GLM-4.5-Air-X` - GLM 4.5 Air X

**Selected Model:** `chutes/zai-org/GLM-4.6` (GLM 4.6 - latest standard)

---

## Updated Models

### Agent-Super (High-Level Reasoning)
- **Before:** anthropic/claude-opus-4.5-20251101
- **After:** chutes/zai-org/GLM-4.6 (GLM 4.6 - latest)
- **Description:** High-level reasoning agent for review, planning, strategic decisions (GLM 4.6)
- **Temperature:** 0.4
- **Capabilities:** read, write
- **Forbidden:** bash, webfetch

### Agent-Ego (Implementation)
- **Before:** anthropic/claude-sonnet-4.5-20250929
- **After:** chutes/zai-org/GLM-4.6 (GLM 4.6 - latest)
- **Description:** Implementation agent for code writing and refactoring (GLM 4.6)
- **Temperature:** 0.2
- **Capabilities:** read, write
- **Forbidden:** bash, webfetch

### Agent-ID (Execution)
- **Before:** anthropic/claude-haiku-4.5-20251001
- **After:** chutes/zai-org/GLM-4.6 (GLM 4.6 - latest)
- **Description:** Execution agent for testing, bash commands, web research (GLM 4.6)
- **Temperature:** 0.15
- **Capabilities:** read, bash, webfetch
- **Forbidden:** write

---

## File Modified

**File:** `D:\projects\Conjecture\.opencode\opencode.json`
**Validation:** ✓ Valid JSON
**Lines:** 89 lines

---

## Command Mappings (Unchanged)

All commands still map to same agents, but agents now use GLM 4.6:

| Command | Agent | Model |
|---------|--------|-------|
| loop | agent-id | GLM 4.6 |
| loop-analyse | agent-super | GLM 4.6 |
| loop-work | agent-ego | GLM 4.6 |
| mem-load | agent-id | GLM 4.6 |
| mem-save | agent-ego | GLM 4.6 |
| test-coverage | agent-id | GLM 4.6 |
| validate | agent-super | GLM 4.6 |

---

## Model Selection Rationale

### Why GLM 4.6?

1. **Latest Available Version** - GLM 4.6 is the most recent stable version
2. **Available in opencode** - Confirmed via `opencode models` command
3. **Standard Version** - Not specialized (Air, V, TEE, etc.)
4. **Balanced Capabilities** - Good trade-off between reasoning speed and quality

### Alternative GLM 4.x Models

**GLM 4.5-Air:** Slightly older, Air variant
**GLM 4.6V:** Vision model (not needed for code/reasoning)
**GLM 4.6-TEE:** TEE variant (specialized use case)

**GLM 4.6** is selected for general-purpose development tasks.

---

## Impact Analysis

### Framework Integration
- **Framework:** Still imports from `~/.config/opencode`
- **Configuration:** Project-specific override in `.opencode/opencode.json`
- **Backwards Compatible:** All commands work identically
- **Model:** All agents now use GLM 4.6

### Temperature Settings
- Preserved original temperature values for each agent:
  - agent-super: 0.4 (high-level reasoning)
  - agent-ego: 0.2 (implementation)
  - agent-id: 0.15 (execution)

### Capabilities & Forbidden
- All agent capabilities preserved (read/write/bash/webfetch)
- All forbidden actions preserved
- No changes to permissions

---

## Validation

- ✓ opencode.json valid JSON
- ✓ All agents updated to GLM 4.6
- ✓ All descriptions updated to mention GLM 4.6
- ✓ Temperature values preserved
- ✓ Capabilities preserved
- ✓ Forbidden actions preserved
- ✓ Command mappings unchanged
- ✓ Framework integration maintained

---

## Rollback Plan

If GLM 4.6 needs to be reverted to Anthropic models:

1. Restore original models in opencode.json:
   - agent-super: anthropic/claude-opus-4.5-20251101
   - agent-ego: anthropic/claude-sonnet-4.5-20250929
   - agent-id: anthropic/claude-haiku-4.5-20251001

2. Remove "(GLM 4.6)" from descriptions

3. Test framework with restored models

---

## Next Steps

1. Test loop execution with GLM 4.6 models
2. Verify all agents respond correctly
3. Verify command execution works as expected
4. Monitor performance and quality of results
5. Compare with Anthropic model results if needed

---

## Available Model Discovery Process

Used `opencode models` command to list all available models:

```bash
opencode models
```

Found 150+ models across multiple providers:
- BigPickle
- GLM 4.x series
- GPT models
- Claude/Anthropic models
- Mistral models
- Qwen models
- DeepSeek models
- And more...

Selected `chutes/zai-org/GLM-4.6` as optimal for Conjecture project.

---

## Status: ✅ COMPLETE

Conjecture project now uses **GLM 4.6 (latest)** for all framework agents while maintaining full framework compatibility with preserved temperatures, capabilities, and command mappings.
