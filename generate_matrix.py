#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Model Matrix quality benchmark
"""

import os
import sys

# Enforce UTF-8 encoding globally
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform.startswith('win'):
    import locale
    try:
        # Try to set UTF-8 encoding on Windows
        os.system('chcp 65001 >nul 2>&1')
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except:
        # Fallback: at least ensure Python handles UTF-8 internally
        pass

from tests.test_model_matrix_simple import ModelMatrixTestSuite

def generate_matrix():
    """Generate Model Matrix with quality scores"""
    suite = ModelMatrixTestSuite()
    matrix = {}

    for harness in suite.harnesses:
        matrix[harness] = {}
        for model in suite.models:
            response = suite.simulate_model_response(harness, model, 'Benchmark test')
            quality = suite.calculate_quality_score(response)
            matrix[harness][model] = round(quality, 1)

    # Calculate averages
    for harness in matrix:
        avg = sum(matrix[harness].values()) / len(matrix[harness])
        matrix[harness]['ROW_AVG'] = round(avg, 1)

    # Column averages
    col_avgs = {}
    for model in suite.models:
        col_avg = (matrix['Direct'][model] + matrix['Conjecture'][model]) / 2
        col_avgs[model] = round(col_avg, 1)

    # Total average
    total_avg = (matrix['Direct']['ROW_AVG'] + matrix['Conjecture']['ROW_AVG']) / 2
    col_avgs['ROW_AVG'] = round(total_avg, 1)

    return matrix, col_avgs

def print_matrix(matrix, col_avgs):
    """Print formatted Model Matrix"""
    print('=== MODEL MATRIX QUALITY BENCHMARK ===')
    print()
    print('               | GraniteTiny | qwen3-4b | GLM-z-9b | GLM-4.6 | ROW_AVG')
    print('-' * 65)
    for harness in ['Direct', 'Conjecture']:
        row = f'{harness:13s} |'
        for model in ['GraniteTiny', 'qwen3-4b', 'GLM-z-9b', 'GLM-4.6', 'ROW_AVG']:
            row += f' {matrix[harness][model]:10.1f} |'
        print(row)
    print('-' * 65)
    row = 'COL_AVG        |'
    for model in ['GraniteTiny', 'qwen3-4b', 'GLM-z-9b', 'GLM-4.6', 'ROW_AVG']:
        row += f' {col_avgs[model]:10.1f} |'
    print(row)

def analyze_insights(matrix, col_avgs):
    """Generate key insights from matrix"""
    print()
    print('Key Insights:')

    # Conjecture improvement
    improvement = ((matrix["Conjecture"]["ROW_AVG"] / matrix["Direct"]["ROW_AVG"] - 1) * 100)
    print(f'- Conjecture improvement: {improvement:.1f}% average improvement')

    # Best harness
    best_harness = "Conjecture" if matrix["Conjecture"]["ROW_AVG"] > matrix["Direct"]["ROW_AVG"] else "Direct"
    print(f'- Best performing harness: {best_harness}')

    # Best model
    best_model = max(["GraniteTiny", "qwen3-4b", "GLM-z-9b", "GLM-4.6"], key=lambda m: col_avgs[m])
    print(f'- Best model: {best_model}')

    # Overall average
    print(f'- Overall quality average: {col_avgs["ROW_AVG"]:.1f}/100')

if __name__ == "__main__":
    matrix, col_avgs = generate_matrix()
    print_matrix(matrix, col_avgs)
    analyze_insights(matrix, col_avgs)