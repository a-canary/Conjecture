# CHOICES.md — Source of Plan

All project choices in priority order. Higher choices constrain lower ones.
Use `/director:choose` to add, change, remove, or reorder choices.
Use `/director:choose-audit` to check for contradictions and structural issues.

## Rules

- **Position = Priority**: higher constrains lower, no exceptions
- **Gravity rule**: changing a choice triggers cascading review
- **Section order is fixed**: Mission > User Experiences > Features > Operations > Data > Architecture > Technology > Implementation
- **Supports line required**: every choice (except top) lists IDs it directly supports
- **Architecture is tool-agnostic**: Architecture describes patterns; Technology names tools
- **No status values**: git diff is the change record

### Choice Entry Format

```
### X-0001: Title of choice
Supports: X-0000, Y-0000

One to two lines of rationale. Not a spec. Just why this choice was made.
```

- ID format: `PREFIX-NNNN` (globally unique 4-digit number)
- Prefixes: `M-` (Mission), `UX-` (User Experiences), `F-` (Features), `O-` (Operations), `D-` (Data), `A-` (Architecture), `T-` (Technology), `I-` (Implementation)

---

## Mission
<!-- Why we exist, values, principles. Everything else flows from here. -->

---

## User Experiences
<!-- Core user journeys, interaction patterns. -->

---

## Features
<!-- Specific capabilities and behaviors. -->

---

## Operations
<!-- Automation vs human-in-loop, workflows, SLAs, moderation. -->

---

## Data
<!-- Data model, schemas, storage decisions. -->

---

## Architecture
<!-- System design, module structure, communication patterns. Tool-agnostic. -->

---

## Technology
<!-- Tech stack, libraries, infrastructure. Names specific tools. -->

---

## Implementation
<!-- Dev practices, coding standards, deployment patterns. -->
