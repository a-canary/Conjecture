# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Test fixtures for Conjecture benchmark tests.
"""

from tests.fixtures.openrouter_free import (
    OpenRouterFreeClient,
    OpenRouterFreeConfig,
    OpenRouterError,
    get_openrouter_api_key,
    openrouter_api_key,
    openrouter_config,
    openrouter_client,
    free_models,
    gpt_oss_20b,
    nemotron_30b,
    benchmark_prompt_factory,
)

__all__ = [
    "OpenRouterFreeClient",
    "OpenRouterFreeConfig",
    "OpenRouterError",
    "get_openrouter_api_key",
    "openrouter_api_key",
    "openrouter_config",
    "openrouter_client",
    "free_models",
    "gpt_oss_20b",
    "nemotron_30b",
    "benchmark_prompt_factory",
]
