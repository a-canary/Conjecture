# DEPRECATED - Advanced Skill Frameworks

**Status:** This document has been removed as part of the Conjecture project simplification.

**Replaced By:** Simple skill templates as described in `requirements.md` and `design.md`

---

## Why This Was Removed

The sophisticated skill frameworks were over-engineered and violated the core principles of the simplified Conjecture design:

- **Too Complex:** Required extensive configuration and understanding
- **Enterprise-Style:** Added unnecessary layers of abstraction
- **Hard to Maintain:** Complex interdependencies and evaluation systems
- **User-Unfriendly:** Required deep knowledge to even understand the system

## Current Simple Approach

The project now uses simple, clear skill templates:

```
Research Skill: Search → Read → Create Claims → Support Evidence
WriteCode Skill: Requirements → Design → Write → Test → Document  
TestCode Skill: Plan Tests → Execute → Fix → Validate
EndClaimEval Skill: Review Evidence → Check Logic → Update Confidence
```

This approach is:
- **Easy to understand** - New team members grasp it in minutes
- **Practical** - Focuses on getting work done
- **Maintainable** - Simple to modify and extend
- **Flexible** - Adapts to different problems without complexity

---

**Last Updated:** November 9, 2025
**Action:** This file can be safely deleted.