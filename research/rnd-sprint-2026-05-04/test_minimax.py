#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""Test MiniMax API connectivity"""
import os
import subprocess
import sys

# Add project to path
sys.path.insert(0, '.')

os.environ['MINIMAX_API_KEY'] = os.environ.get('MINIMAX_API_KEY') or subprocess.run(
    ['pass', 'show', 'minimax'], capture_output=True, text=True
).stdout.strip()

from openai import OpenAI

client = OpenAI(
    api_key=os.environ['MINIMAX_API_KEY'],
    base_url='https://api.minimax.io/v1'
)

try:
    response = client.chat.completions.create(
        model='MiniMax-M2.7',
        messages=[{'role': 'user', 'content': 'What is 2+2? Respond with just the number.'}],
        max_tokens=20,
        temperature=0.1
    )
    print(f'SUCCESS! Response: {response.choices[0].message.content}')
    print(f'Model: {response.model}')
except Exception as e:
    print(f'Error: {type(e).__name__}: {e}')