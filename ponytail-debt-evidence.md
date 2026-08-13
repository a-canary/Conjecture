# ponytail-debt: conjecture

Task: 000244-hygiene-conjecture-ponytail-debt

## Summary

Scanned conjecture repo for `ponytail:` annotations — inline developer markers
noting edge cases, design decisions, and architectural watch-items.

**7 markers found** across 5 files. Categorized into:

- **1 resolved inline** (removed dead `if __name__` block, tests already cover the path)
- **4 clarify-docs** hygiene tasks emitted
- **1 improve-architecture** hygiene task emitted

## Markers Found

### src/utils/id_utils.py (1 — RESOLVED)
- L28: `ponytail: smallest self-check that fails if the helper breaks.`
- **Resolution:** Removed the dead `if __name__ == "__main__":` block. The `generate_id`
  function is already exercised by `tests/test_id_utilities.py` (new test methods added
  covering format, prefix, length validation, custom length, and uniqueness).
- **Commit:** included in this task's diff.

### benchmarks/deepeval_suite.py (3)
- L1504: `ponytail: installed deepeval 2.6.7 loads the retired bare "gsm8k" HF id...`
  → **clarify-docs**: Monkey-patch workaround for deepeval 2.6.7 loading the retired
    bare `"gsm8k"` HF dataset id. Should be documented as known dependency quirk in
    docs/reference/setup.md so maintainers know to drop it on deepeval upgrade.

- L1633: `ponytail: single-benchmark proxy; a labelled multi-benchmark set replaces...`
  → **improve-architecture**: `compute_paired_verdict` hardcodes single-benchmark
    assumptions (router_accuracy = share classified math for GSM8K). Architecture
    should support labelled multi-benchmark sets before adding new benchmarks.

- L1847: `ponytail: minimal .generate() shim, only interface _call_model needs`
  → **clarify-docs**: `_DirectArmModel` is an ad-hoc shim for the direct (non-routed)
    arm in paired evaluation. Its contract (single-method `.generate()`, no state,
    wraps raw OpenAI client) is implicit. Should document the interface contract.

### benchmarks/paired_stats.py (1)
- L30: `ponytail: normal approx; exact McNemar interval if n_clean ever stays <30.`
  → **clarify-docs**: `paired_delta` uses normal approximation for the CI of paired
    differences. The ponytail notes an exact McNemar interval should replace it if
    `n_clean` ever drops below 30. Should document the statistical methodology choice
    and the switching condition.

### src/endpoint/conjecture_endpoint.py (1)
- L610: `ponytail: lazy import keeps load order safe between endpoint + resumption`
  → **clarify-docs**: Lazy import of `resume_evaluation` inside the method body
    avoids a circular/load-order hazard between `conjecture_endpoint` and `resumption`.
    Should document the inter-module dependency and why the lazy import is necessary.

### src/endpoint/resumption.py (1)
- L29: `ponytail: re-anchor the logger to the parent module's name so log filters...`
  → **clarify-docs**: After the `resumption` module was extracted from
    `conjecture_endpoint`, the logger was re-anchored to
    `src.endpoint.conjecture_endpoint` so existing dashboards/log filters keep
    matching. Should document the logging namespace decision in resumption.py's
    module docstring.

## Hygiene Tasks Emitted

| Skill | Title | Description |
|-------|-------|-------------|
| clarify-docs | Document deepeval GSM8K dataset monkey-patch | deepeval 2.6.7 loads retired bare "gsm8k" HF id; monkey-patched to "openai/gsm8k" — document in docs/reference/setup.md with drop-after-upgrade note |
| improve-architecture | Generalize benchmark runner for multi-benchmark sets | compute_paired_verdict hardcodes GSM8K-only router_accuracy heuristic; architecture should support labelled multi-benchmark sets |
| clarify-docs | Document _DirectArmModel interface contract | Ad-hoc .generate() shim for direct-arm paired evaluation; implicit contract should be documented |
| clarify-docs | Document paired_stats normal approximation methodology | paired_delta uses normal approx CI; exact McNemar interval should replace if n_clean < 30 — document switching condition |
| clarify-docs | Document lazy import in conjecture_endpoint.resume_evaluation | Load-order dependency between endpoint + resumption modules; lazy import avoids circular hazard |
