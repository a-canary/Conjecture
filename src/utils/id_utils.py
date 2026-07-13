# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
ID generation utilities.

Single source of truth for the `f"{prefix}{uuid4().hex[:length]}"` pattern
used across session, chat-completion, and sub-claim identifiers. The
canonical `generate_claim_id` for claim IDs lives in `src.core.models` and
is intentionally separate (it includes a timestamp prefix).
"""

import uuid


def generate_id(prefix: str, length: int = 8) -> str:
    """Return ``f"{prefix}{uuid4().hex[:length]}"``.

    Args:
        prefix: String prepended to the random suffix (e.g. ``"s"`` for sessions).
        length: Number of hex chars from the UUID4 to keep. Defaults to 8.
    """
    if length < 1:
        raise ValueError(f"length must be >= 1, got {length}")
    return f"{prefix}{uuid.uuid4().hex[:length]}"


if __name__ == "__main__":
    # ponytail: smallest self-check that fails if the helper breaks.
    sample = generate_id("s")
    assert sample.startswith("s")
    assert len(sample) == 9
    assert int(sample[1:], 16) >= 0
    long = generate_id("chatcmpl-")
    assert long.startswith("chatcmpl-") and len(long) == len("chatcmpl-") + 8
    print("ok")
