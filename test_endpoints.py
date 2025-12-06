#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test provider endpoints directly
"""

import os
import sys
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from processing.llm.openai_compatible_provider import OpenAICompatibleProcessor

    # Test granite-4-h-tiny (local)
    print('[*] Testing granite-4-h-tiny endpoint...')
    config = {
        'url': 'http://localhost:1234',
        'api_key': '',
        'model': 'ibm/granite-4-h-tiny'
    }
    try:
        processor = OpenAICompatibleProcessor(config)
        response = processor.generate_response('Hello, can you count to 3?')
        print(f'[OK] granite-4-h-tiny: {response[:100]}...')
    except Exception as e:
        print(f'[FAIL] granite-4-h-tiny: {e}')

    # Test GLM-4.6 (cloud)
    print('\n[*] Testing GLM-4.6 endpoint...')
    config = {
        'url': 'https://api.z.ai/api/coding/paas/v4',
        'api_key': '70e6e12e4d7c46e2a4d0b85503d51f38.LQHl8d98kDJChttb',
        'model': 'glm-4.6'
    }
    try:
        processor = OpenAICompatibleProcessor(config)
        response = processor.generate_response('Hello, can you count to 3?')
        print(f'[OK] GLM-4.6: {response[:100]}...')
    except Exception as e:
        print(f'[FAIL] GLM-4.6: {e}')

    # Test GPT-OSS-20b (cloud)
    print('\n[*] Testing GPT-OSS-20b endpoint...')
    config = {
        'url': 'https://openrouter.ai/api/v1',
        'api_key': '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',
        'model': 'openai/gpt-oss-20b'
    }
    try:
        processor = OpenAICompatibleProcessor(config)
        response = processor.generate_response('Hello, can you count to 3?')
        print(f'[OK] GPT-OSS-20b: {response[:100]}...')
    except Exception as e:
        print(f'[FAIL] GPT-OSS-20b: {e}')

    print('\n[SUCCESS] Provider endpoint validation complete!')

except Exception as e:
    print(f'[ERROR] Could not test endpoints: {e}')
    import traceback
    traceback.print_exc()