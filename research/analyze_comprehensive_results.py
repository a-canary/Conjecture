#!/usr/bin/env python3
"""
Comprehensive Analysis Report Generator
Analyzes reasoning and agentic capabilities across all models
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import statistics

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def load_results():
    """Load the most recent comprehensive experiment results"""
    results_dir = Path("research/results")

    # Find the comprehensive experiment file
    json_files = list(results_dir.glob("conjecture_comprehensive_*.json"))

    if not json_files:
        print("No comprehensive experiment results found!")
        return None

    latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
    print(f"Loading results from: {latest_file}")

    with open(latest_file, 'r') as f:
        data = json.load(f)

    return data

def analyze_reasoning_capabilities(results: List[Dict[str, Any]]):
    """Analyze reasoning capabilities by model and approach"""
    print("\n" + "=" * 80)
    print("REASONING CAPABILITIES ANALYSIS")
    print("=" * 80)

    # Group by model and approach
    grouped = {}
    for result in results:
        if result["status"] != "success":
            continue

        key = (result["model"], result["approach"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(result)

    analysis = {}

    for (model, approach), result_list in grouped.items():
        print(f"\n{model} | {approach}:")
        print("-" * 80)

        # Calculate metrics
        avg_reasoning_steps = statistics.mean([r["reasoning_steps"] for r in result_list])
        avg_response_length = statistics.mean([r["response_length"] for r in result_list])
        avg_self_consistency = statistics.mean([r["self_consistency_score"] for r in result_list])

        # Claim generation rate
        claim_success_rate = sum(1 for r in result_list if r["has_claim_format"]) / len(result_list) * 100

        # Quality scores (if available)
        quality_metrics = {}
        for metric in ["correctness_score", "reasoning_quality_score", "coherence_score", "depth_score"]:
            scores = [r[metric] for r in result_list if r.get(metric) is not None]
            if scores:
                quality_metrics[metric] = statistics.mean(scores)

        print(f"  Tests: {len(result_list)}")
        print(f"  Avg reasoning steps: {avg_reasoning_steps:.1f}")
        print(f"  Avg response length: {avg_response_length:.0f} chars")
        print(f"  Self-consistency: {avg_self_consistency:.2f}")
        print(f"  Claim generation: {claim_success_rate:.0f}%")

        if quality_metrics:
            print(f"  Quality scores:")
            for metric, score in quality_metrics.items():
                print(f"    {metric}: {score:.2f}")

        analysis[f"{model} | {approach}"] = {
            "model": model,
            "approach": approach,
            "test_count": len(result_list),
            "avg_reasoning_steps": avg_reasoning_steps,
            "avg_response_length": avg_response_length,
            "claim_success_rate": claim_success_rate,
            "quality_metrics": quality_metrics
        }

    return analysis

def analyze_agentic_capabilities(results: List[Dict[str, Any]]):
    """Analyze agentic capabilities (planning, task decomposition, etc.)"""
    print("\n" + "=" * 80)
    print("AGENTIC CAPABILITIES ANALYSIS")
    print("=" * 80)

    # Focus on planning and code analysis tasks
    agentic_tasks = ["planning_multi_step", "code_analysis_debug"]
    agentic_results = [r for r in results if r["status"] == "success" and r["test_case_id"] in agentic_tasks]

    if not agentic_results:
        print("No agentic task results available")
        return {}

    grouped = {}
    for result in agentic_results:
        key = (result["model"], result["approach"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(result)

    analysis = {}

    for (model, approach), result_list in grouped.items():
        print(f"\n{model} | {approach}:")
        print("-" * 80)

        avg_reasoning_steps = statistics.mean([r["reasoning_steps"] for r in result_list])
        avg_response_length = statistics.mean([r["response_length"] for r in result_list])

        # Agentic capability score (if available)
        agentic_scores = [r["agentic_capability_score"] for r in result_list if r.get("agentic_capability_score") is not None]
        avg_agentic = statistics.mean(agentic_scores) if agentic_scores else None

        print(f"  Agentic tasks: {len(result_list)}")
        print(f"  Avg reasoning steps: {avg_reasoning_steps:.1f}")
        print(f"  Avg response length: {avg_response_length:.0f} chars")

        if avg_agentic:
            print(f"  Agentic capability score: {avg_agentic:.2f}")

        analysis[f"{model} | {approach}"] = {
            "model": model,
            "approach": approach,
            "task_count": len(result_list),
            "avg_reasoning_steps": avg_reasoning_steps,
            "avg_response_length": avg_response_length,
            "avg_agentic_score": avg_agentic
        }

    return analysis

def evaluate_conjecture_hypothesis(analysis: Dict[str, Any]):
    """Evaluate the main hypothesis: Does Conjecture enable tiny LLMs to match SOTA?"""
    print("\n" + "=" * 80)
    print("CONJECTURE HYPOTHESIS EVALUATION")
    print("=" * 80)
    print("Hypothesis: Conjecture methods enable tiny LLMs to perform near SOTA")
    print("            on reasoning and agentic coding tasks")
    print("=" * 80)

    # Compare tiny vs SOTA models
    tiny_results = {k: v for k, v in analysis.items() if "tiny" in v["model"]}
    sota_results = {k: v for k, v in analysis.items() if "sota" in v["model"]}

    print(f"\nMODEL COMPARISON:")
    print(f"Tiny models tested: {len(tiny_results)}")
    print(f"SOTA models tested: {len(sota_results)}")

    if not tiny_results or not sota_results:
        print("\n[WARNING] Cannot evaluate hypothesis - missing model comparisons")
        return

    # Compare reasoning steps (proxy for reasoning depth)
    print(f"\nREASONING DEPTH:")
    for model_key, data in tiny_results.items():
        approach = data["approach"]
        tiny_steps = data["avg_reasoning_steps"]

        # Find corresponding SOTA result
        sota_key = None
        for k, v in sota_results.items():
            if v["approach"] == approach:
                sota_key = k
                break

        if sota_key:
            sota_steps = sota_results[sota_key]["avg_reasoning_steps"]
            ratio = tiny_steps / sota_steps if sota_steps > 0 else 0

            print(f"\n  {model_key}:")
            print(f"    Tiny LLM: {tiny_steps:.1f} reasoning steps")
            print(f"    SOTA: {sota_steps:.1f} reasoning steps")
            print(f"    Ratio: {ratio:.2f} ({ratio*100:.0f}% of SOTA)")

            if ratio >= 0.8:
                print(f"    [SUCCESS] Near SOTA performance (>=80%)")
            elif ratio >= 0.6:
                print(f"    [WARNING] Moderate performance (60-80%)")
            else:
                print(f"    [FAIL] Below SOTA performance (<60%)")

    # Compare claim generation (Conjecture-specific)
    print(f"\nCLAIM GENERATION:")
    conjecture_results = {k: v for k, v in analysis.items() if "true_conjecture" in k}

    for model_key, data in conjecture_results.items():
        claim_rate = data["claim_success_rate"]
        print(f"\n  {model_key}:")
        print(f"    Claim generation success: {claim_rate:.0f}%")

        if claim_rate >= 80:
            print(f"    [EXCELLENT] Claim generation")
        elif claim_rate >= 60:
            print(f"    [GOOD] Claim generation")
        else:
            print(f"    [POOR] Claim generation")

    # Agentic capabilities
    print(f"\nAGENTIC CAPABILITIES:")
    # Note: Would need agentic scores to properly evaluate
    print(f"    Agentic tasks tested: planning, code analysis")
    print(f"    Analysis shows reasoning steps and response length")

def generate_recommendations(analysis: Dict[str, Any]):
    """Generate recommendations based on findings"""
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    print("""
1. INCREASE TEST DIVERSITY:
   - Add more complex reasoning tasks
   - Include multi-step coding challenges
   - Test with real-world agent scenarios

2. IMPROVE EVALUATION:
   - Implement ground truth comparison
   - Add human expert evaluation
   - Create domain-specific benchmarks

3. OPTIMIZE CONJECTURE:
   - Simplify claim format for better compliance
   - Add claim validation and correction
   - Implement adaptive claim count based on complexity

4. EXPAND MODEL TESTING:
   - Test more tiny models (<3B parameters)
   - Include intermediate sizes (4-7B)
   - Compare against more SOTA models

5. FOCUS ON QUALITY OVER SPEED:
   - Accuracy is more important than response time
   - Measure reasoning correctness
   - Evaluate agentic task completion rates
""")

def generate_report(data: Dict[str, Any]):
    """Generate comprehensive report"""
    print("COMPREHENSIVE CONJECTURE RESEARCH REPORT")
    print("=" * 80)

    results = data.get("results", [])
    timestamp = data.get("timestamp", "")

    print(f"\nExperiment ID: {data.get('experiment_id')}")
    print(f"Timestamp: {timestamp}")
    print(f"Models tested: {len(data.get('models_tested', []))}")
    print(f"Test cases: {len(data.get('test_cases', []))}")
    print(f"Total tests: {len(results)}")

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]

    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    # Analyze reasoning capabilities
    reasoning_analysis = analyze_reasoning_capabilities(successful)

    # Analyze agentic capabilities
    agentic_analysis = analyze_agentic_capabilities(successful)

    # Evaluate hypothesis
    evaluate_conjecture_hypothesis(reasoning_analysis)

    # Generate recommendations
    generate_recommendations(reasoning_analysis)

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("""
The comprehensive experiment successfully tested reasoning and agentic capabilities
across multiple models using scientific methods. Key achievements:

✅ Model-by-model testing (prevents LM Studio reloading)
✅ Comprehensive metrics (reasoning steps, claim generation, self-consistency)
✅ Diverse test cases (6 categories of reasoning/agentic tasks)
✅ Quality evaluation framework (LLM-as-a-Judge ready)

The hypothesis that Conjecture enables tiny LLMs to match SOTA performance
requires further testing with:
- More model comparisons (tiny vs SOTA)
- Accuracy/quality metrics
- Complex agentic tasks
- Ground truth evaluation
""")
    print("=" * 80)

def main():
    """Main analysis function"""
    print("COMPREHENSIVE CONJECTURE RESEARCH ANALYSIS")
    print("Analyzing reasoning and agentic capabilities")

    data = load_results()
    if not data:
        print("No data to analyze!")
        return

    generate_report(data)

    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = Path(f"research/results/comprehensive_report_{timestamp}.md")

    # Capture printed output to file
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    sys.stdout = buffer = StringIO()

    generate_report(data)

    sys.stdout = old_stdout
    report_content = buffer.getvalue()

    with open(report_file, 'w') as f:
        f.write(report_content)

    print(f"\n[OK] Report saved to: {report_file}")

if __name__ == "__main__":
    main()
