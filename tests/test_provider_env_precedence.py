# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Test for provider API-key resolution (env_var > api field > legacy key field).
Targets the env-var-first load path introduced when scrubbing hardcoded
API keys from .conjecture/config.json.
"""

import ast
import os
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
TARGET = REPO_ROOT / "src" / "processing" / "simplified_llm_manager.py"


@pytest.fixture(scope="module")
def resolver():
    """Extract `_resolve_provider_secrets` from the source file as a
    standalone function and return it. Avoids importing the full LLM
    manager (and its OpenAI provider stack) just to test a pure resolver."""
    src = TARGET.read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "_resolve_provider_secrets"
            and any(
                isinstance(d, ast.Name) and d.id == "staticmethod"
                for d in node.decorator_list
            )
        ):
            snippet = ast.get_source_segment(src, node)
            # Inline-import typing for the annotations so the snippet runs
            # without the original module's import block.
            prelude = "from typing import Any, Dict\n"
            code = textwrap.dedent(prelude + snippet)
            namespace = {"os": os}
            exec(code, namespace)
            return namespace["_resolve_provider_secrets"]
    raise RuntimeError("_resolve_provider_secrets not found in source")


def test_env_var_wins_over_api_field(resolver):
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("OPENROUTER_API_KEY", "sk-or-v1-FROM-ENV")
        result = resolver(
            {
                "name": "openrouter",
                "url": "https://openrouter.ai/api/v1",
                "api": "sk-or-v1-LEAKED-IN-FILE",
                "env_var": "OPENROUTER_API_KEY",
                "priority": 1,
            },
            0,
        )
    assert result["api"] == "sk-or-v1-FROM-ENV"
    assert result["env_var"] == "OPENROUTER_API_KEY"


def test_api_field_used_when_env_unset(resolver):
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("OPENROUTER_API_KEY", raising=False)
        result = resolver(
            {
                "name": "openrouter",
                "url": "https://openrouter.ai/api/v1",
                "api": "sk-or-v1-INLINE",
                "env_var": "OPENROUTER_API_KEY",
                "priority": 1,
            },
            0,
        )
    assert result["api"] == "sk-or-v1-INLINE"


def test_local_provider_with_no_env_var(resolver):
    """Local providers (LM Studio) carry a placeholder api field and no env_var."""
    result = resolver(
        {
            "name": "lm_studio",
            "url": "http://localhost:1234/v1",
            "api": "not-needed",
            "priority": 3,
        },
        2,
    )
    assert result["api"] == "not-needed"
    assert result["env_var"] is None


def test_empty_env_falls_through_to_api_field(resolver):
    """An empty env var must NOT clobber a valid inline api field."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("OPENROUTER_API_KEY", "")
        result = resolver(
            {
                "name": "openrouter",
                "url": "https://openrouter.ai/api/v1",
                "api": "sk-or-v1-INLINE",
                "env_var": "OPENROUTER_API_KEY",
            },
            0,
        )
    assert result["api"] == "sk-or-v1-INLINE"


def test_legacy_key_field_still_supported(resolver):
    """Pre-2024 configs used 'key' instead of 'api' — must not regress."""
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("OPENROUTER_API_KEY", raising=False)
        result = resolver(
            {"name": "foo", "url": "https://x", "key": "legacy-key"},
            0,
        )
    assert result["api"] == "legacy-key"


def test_missing_name_falls_back_to_index(resolver):
    """Bare configs without a name field get a synthetic provider_N name."""
    result = resolver({"url": "https://x"}, 7)
    assert result["name"] == "provider_7"
    assert result["api"] == ""