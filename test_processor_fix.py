#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test processor configuration fix
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    print('[*] Testing processor initialization...')

    from processing.llm.openai_compatible_provider import OpenAICompatibleProcessor

    # Test granite-4-h-tiny configuration
    print('[*] Testing granite-4-h-tiny...')
    processor = OpenAICompatibleProcessor(
        api_key='',
        api_url='http://localhost:1234',
        model_name='ibm/granite-4-h-tiny',
        provider_name='granite-4-h-tiny'
    )
    print(f'[OK] granite-4-h-tiny processor created: {processor.api_url}')

    # Test a simple call
    print('[*] Testing simple API call...')
    result = processor.generate_response('Hello, count to 3')
    print(f'[OK] Response: {result.response[:100]}...')

    print('[SUCCESS] Processor configuration fixed!')

except Exception as e:
    print(f'[ERROR] {e}')
    import traceback
    traceback.print_exc()