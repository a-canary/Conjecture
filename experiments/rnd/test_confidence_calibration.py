#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
E2: Confidence Calibration Validation

Validates the Hierarchical Confidence Calibration Algorithm (HCCA):
C = 0.19×Local + 0.28×Direct + 0.30×Transitive + 0.23×Prior  # E11 optimized

Verification: calibration_error_reduction > 0.3 after 2 iterations
"""

import json
import os
import sys
import random
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import deque

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.core.models import Claim, ClaimType, ClaimState, DirtyReason, ClaimScope
from src.core.claim_operations import (
    update_confidence,
    find_sub_claims,
    batch_update_confidence,
)


# =============================================================================
# HCCA IMPLEMENTATION
# =============================================================================

@dataclass
class CalibrationWeights:
    """Weights for the HCCA formula."""
    local: float = 0.19  # E11 optimized
    direct: float = 0.28  # E11 optimized
    transitive: float = 0.30  # E11 optimized
    prior: float = 0.23  # E11 optimized


@dataclass
class CalibrationReport:
    """Report from a single calibration operation."""
    claim_id: str
    old_confidence: float
    new_confidence: float
    direct_support: Tuple[float, int]  # (avg_confidence, count)
    transitive_evidence: float
    type_prior: float
    weights_used: CalibrationWeights
    calibration_timestamp: datetime
    changed: bool


# Claim type priors
CLAIM_TYPE_PRIORS = {
    ClaimType.IMPRESSION: 0.35,
    ClaimType.ASSUMPTION: 0.40,
    ClaimType.OBSERVATION: 0.55,
    ClaimType.CONJECTURE: 0.40,
    ClaimType.CONCEPT: 0.50,
    ClaimType.EXAMPLE: 0.60,
    ClaimType.GOAL: 0.45,
    ClaimType.REFERENCE: 0.50,
    ClaimType.ASSERTION: 0.65,
}

DEFAULT_WEIGHTS = CalibrationWeights()
MAX_STEP = 0.15  # Max 15% change per calibration


def get_type_prior(claim: Claim) -> float:
    """Get Bayesian prior for a claim based on its type."""
    if not claim.type:
        return 0.5
    primary_type = claim.type[0]
    return CLAIM_TYPE_PRIORS.get(primary_type, 0.5)


def calculate_transitive_evidence(
    claim: Claim,
    all_claims: List[Claim],
    max_depth: int = 5,
    decay_factor: float = 0.7,
) -> float:
    """Calculate transitive evidence using depth-weighted BFS."""
    # Build sub_map for efficient lookup
    sub_map: Dict[str, Set[str]] = {}
    claim_map: Dict[str, Claim] = {}
    
    for c in all_claims:
        claim_map[c.id] = c
        sub_map[c.id] = set(c.subs)
    
    if claim.id not in sub_map:
        return 0.0
    
    weighted_sum = 0.0
    weight_total = 0.0
    
    # BFS with depth tracking
    queue = deque([(claim.id, 0)])
    visited = {claim.id}
    
    while queue:
        current_id, depth = queue.popleft()
        
        if depth > max_depth:
            continue
        
        for sub_id in sub_map.get(current_id, set()):
            if sub_id in visited:
                continue
            visited.add(sub_id)
            
            sub_claim = claim_map.get(sub_id)
            if not sub_claim:
                continue
            
            depth_weight = decay_factor ** depth
            weighted_sum += sub_claim.confidence * depth_weight
            weight_total += depth_weight
            
            queue.append((sub_id, depth + 1))
    
    return weighted_sum / weight_total if weight_total > 0 else 0.0


def smooth_confidence_change(old_confidence: float, new_calibration: float, max_step: float = MAX_STEP) -> float:
    """Apply smoothing to prevent extreme jumps."""
    diff = new_calibration - old_confidence
    
    if abs(diff) <= max_step:
        return new_calibration
    
    # Clamp to max step
    if diff > 0:
        return old_confidence + max_step
    else:
        return old_confidence - max_step


def calibrate_confidence(
    claim: Claim,
    all_claims: List[Claim],
    weights: CalibrationWeights = DEFAULT_WEIGHTS,
    decay_factor: float = 0.7,
    min_confidence_change: float = 0.01,
) -> Tuple[Claim, CalibrationReport]:
    """Calibrate a claim's confidence based on its evidence hierarchy."""
    
    # Step 1: Collect direct sub-claims
    direct_sub_claims = find_sub_claims(claim, all_claims)
    direct_count = len(direct_sub_claims)
    
    # Step 2: Calculate Direct Support Score
    if direct_count > 0:
        direct_support = sum(c.confidence for c in direct_sub_claims) / direct_count
    else:
        direct_support = 0.0
    
    # Step 3: Calculate Transitive Evidence Score
    transitive_evidence = calculate_transitive_evidence(claim, all_claims, max_depth=5, decay_factor=decay_factor)
    
    # Step 4: Calculate Bayesian Prior based on claim type
    type_prior = get_type_prior(claim)
    
    # Step 5: Compute weighted calibration
    raw_calibrated = (
        weights.local * claim.confidence +
        weights.direct * direct_support +
        weights.transitive * transitive_evidence +
        weights.prior * type_prior
    )
    
    # Step 6: Apply smoothing
    smoothed = smooth_confidence_change(
        old_confidence=claim.confidence,
        new_calibration=raw_calibrated,
        max_step=MAX_STEP,
    )
    
    # Step 7: Clamp to valid range
    final_confidence = max(0.0, min(1.0, smoothed))
    
    # Step 8: Generate report
    report = CalibrationReport(
        claim_id=claim.id,
        old_confidence=claim.confidence,
        new_confidence=final_confidence,
        direct_support=(direct_support, direct_count),
        transitive_evidence=transitive_evidence,
        type_prior=type_prior,
        weights_used=weights,
        calibration_timestamp=datetime.now(timezone.utc),
        changed=abs(final_confidence - claim.confidence) >= min_confidence_change,
    )
    
    # Step 9: Create updated claim
    updated_claim = update_confidence(claim, final_confidence)
    
    return updated_claim, report


def batch_calibrate(
    claims: List[Claim],
    weights: CalibrationWeights = DEFAULT_WEIGHTS,
    max_iterations: int = 3,
    decay_factor: float = 0.7,
) -> Tuple[List[Claim], List[Dict[str, Any]]]:
    """Batch calibrate all claims, iterating to allow propagation."""
    claims = list(claims)  # Don't mutate input
    iteration_reports = []
    
    for iteration in range(max_iterations):
        iteration_claims = list(claims)
        changed_count = 0
        total_error_before = 0.0
        total_error_after = 0.0
        
        for claim in iteration_claims:
            old_conf = claim.confidence
            updated, report = calibrate_confidence(
                claim, claims, weights=weights, decay_factor=decay_factor
            )
            
            if report.changed:
                changed_count += 1
                # Update in place for next iteration
                idx = next((i for i, c in enumerate(claims) if c.id == claim.id), None)
                if idx is not None:
                    claims[idx] = updated
                    total_error_after += abs(updated.confidence - old_conf)
        
        iteration_reports.append({
            "iteration": iteration + 1,
            "changed_claims": changed_count,
            "total_error_after": total_error_after,
        })
        
        if changed_count == 0:
            break  # Converged
    
    return claims, iteration_reports


# =============================================================================
# TEST DATA FROM BENCHMARK RESULTS
# =============================================================================

def load_benchmark_claims() -> List[Dict[str, Any]]:
    """Load claims from existing benchmark results."""
    results_dir = "/home/aaron/projects/conjecture/experiments/results"
    
    all_claims = []
    
    for filename in os.listdir(results_dir):
        if not filename.endswith('.json'):
            continue
        
        filepath = os.path.join(results_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Extract accuracy information from benchmark results
            if 'results' in data and isinstance(data['results'], list):
                for item in data['results']:
                    if 'accuracy' in item and 'correct' in item:
                        # This is a benchmark result with known accuracy
                        n_correct = item.get('correct', 0)
                        total = item.get('total', 1)
                        accuracy = item.get('accuracy', 0) / 100.0 if item.get('accuracy', 0) > 1 else item.get('accuracy', 0)
                        
                        claim = Claim(
                            id=f"bench_{len(all_claims)}",
                            content=f"{data.get('experiment', 'unknown')}: {item.get('benchmark', 'unknown')}",
                            confidence=min(0.95, max(0.1, accuracy + 0.1)),  # Initial estimate
                            type=[ClaimType.CONJECTURE],
                            tags=["benchmark", data.get('model', 'unknown')],
                            state=ClaimState.VALIDATED,
                            supers=[],
                            subs=[],
                        )
                        
                        all_claims.append({
                            "claim": claim,
                            "actual_accuracy": accuracy,
                            "correct": n_correct,
                            "total": total,
                            "source": filename,
                        })
        except Exception as e:
            continue
    
    return all_claims


def create_synthetic_test_claims(n_claims: int = 50) -> List[Dict[str, Any]]:
    """Create synthetic test claims with known accuracy for calibration testing."""
    random.seed(123)
    claims_data = []
    
    base_claims = [
        ("This model performs well on math problems.", 0.65, ClaimType.CONJECTURE),
        ("The benchmark shows high accuracy.", 0.72, ClaimType.OBSERVATION),
        ("Model degradation occurs after many iterations.", 0.55, ClaimType.ASSUMPTION),
        ("Chain-of-thought improves reasoning.", 0.70, ClaimType.CONJECTURE),
        ("Larger models generally perform better.", 0.68, ClaimType.ASSUMPTION),
        ("Few-shot prompting helps generalization.", 0.74, ClaimType.OBSERVATION),
        ("Context length affects performance.", 0.62, ClaimType.ASSUMPTION),
        ("Temperature settings change output diversity.", 0.58, ClaimType.OBSERVATION),
        ("Batch processing improves efficiency.", 0.69, ClaimType.ASSERTION),
        ("Memory constraints limit model size.", 0.71, ClaimType.ASSUMPTION),
    ]
    
    for i in range(n_claims):
        base_content, base_acc, base_type = base_claims[i % len(base_claims)]
        
        # Add some noise to create varied claims
        content = f"{base_content} (variant {i})"
        
        # Actual accuracy varies around the base
        actual_accuracy = base_acc + random.uniform(-0.15, 0.15)
        actual_accuracy = max(0.1, min(0.95, actual_accuracy))
        
        # Stated confidence may be off from actual
        stated_confidence = actual_accuracy + random.uniform(-0.2, 0.2)
        stated_confidence = max(0.1, min(0.95, stated_confidence))
        
        # Create claim with stated confidence
        claim = Claim(
            id=f"syn_cal_{i}",
            content=content,
            confidence=stated_confidence,
            type=[base_type],
            tags=["synthetic", "calibration-test"],
            state=ClaimState.VALIDATED,
            supers=[],
            subs=[],
        )
        
        # Add some sub-claims for direct/transitive evidence
        n_subs = random.randint(1, 3)
        for j in range(n_subs):
            sub_conf = actual_accuracy + random.uniform(-0.1, 0.1)
            sub_claim = Claim(
                id=f"syn_cal_{i}_sub_{j}",
                content=f"Supporting evidence {j} for: {content[:50]}",
                confidence=max(0.1, min(0.95, sub_conf)),
                type=[ClaimType.OBSERVATION],
                tags=["synthetic", "sub-claim"],
                state=ClaimState.VALIDATED,
                supers=[f"syn_cal_{i}"],
                subs=[],
            )
            claims_data.append({
                "claim": sub_claim,
                "actual_accuracy": sub_conf,
                "source": "synthetic",
            })
        
        claims_data.append({
            "claim": claim,
            "actual_accuracy": actual_accuracy,
            "source": "synthetic",
        })
    
    return claims_data


# =============================================================================
# MAIN TEST
# =============================================================================

def run_confidence_calibration_experiment():
    """Run the confidence calibration validation experiment."""
    print("=" * 70)
    print("E2: Confidence Calibration Validation")
    print("=" * 70)
    
    # Load or create test claims
    print("\n[1/5] Loading benchmark claims...")
    benchmark_claims = load_benchmark_claims()
    print(f"  Loaded {len(benchmark_claims)} claims from benchmark results")
    
    # If not enough benchmark claims, supplement with synthetic
    if len(benchmark_claims) < 30:
        print("\n[2/5] Supplementing with synthetic claims...")
        synthetic_claims = create_synthetic_test_claims(50)
        all_claims_data = benchmark_claims + synthetic_claims
    else:
        all_claims_data = benchmark_claims[:50]  # Use first 50
    
    print(f"  Total claims for experiment: {len(all_claims_data)}")
    
    # Extract claims and ground truth
    claims_list = [d["claim"] for d in all_claims_data]
    actual_accuracies = {d["claim"].id: d["actual_accuracy"] for d in all_claims_data}
    
    # Add sub-claim relationships for testing transitive evidence
    # Build a hierarchy for batch calibration
    claims_map = {c.id: c for c in claims_list}
    
    # For claims without subs, add synthetic sub-claims
    for i, claim in enumerate(claims_list):
        if not claim.subs and i > 0:
            # Link to previous claim as a sub
            prev_claim = claims_list[i - 1]
            # Create a sub claim
            sub_claim = Claim(
                id=f"cal_sub_{claim.id}",
                content=f"Evidence supporting: {claim.content[:60]}",
                confidence=claim.confidence * 0.9,
                type=[ClaimType.OBSERVATION],
                tags=["calibration", "sub"],
                state=ClaimState.VALIDATED,
                supers=[claim.id],
                subs=[],
            )
            claims_map[sub_claim.id] = sub_claim
            # Update claim to have this sub
            claim.subs.append(sub_claim.id)
    
    claims_list = list(claims_map.values())
    print(f"  Claims with subs for transitive testing: {sum(1 for c in claims_list if c.subs)}")
    
    # Calculate initial calibration error
    print("\n[3/5] Computing initial calibration error...")
    initial_errors = []
    for claim in claims_list:
        if claim.id in actual_accuracies:
            error = abs(claim.confidence - actual_accuracies[claim.id])
            initial_errors.append(error)
    
    initial_mean_error = sum(initial_errors) / len(initial_errors) if initial_errors else 0.5
    print(f"  Initial mean calibration error: {initial_mean_error:.4f}")
    
    # Run iteration 1
    print("\n[4/5] Running HCCA calibration (iteration 1)...")
    claims_iter1, report1 = batch_calibrate(
        list(claims_list),
        weights=DEFAULT_WEIGHTS,
        max_iterations=1,
        decay_factor=0.7,
    )
    
    errors_iter1 = []
    for claim in claims_iter1:
        if claim.id in actual_accuracies:
            error = abs(claim.confidence - actual_accuracies[claim.id])
            errors_iter1.append(error)
    
    mean_error_iter1 = sum(errors_iter1) / len(errors_iter1) if errors_iter1 else 0.5
    error_reduction_1 = initial_mean_error - mean_error_iter1
    print(f"  Mean error after iteration 1: {mean_error_iter1:.4f}")
    print(f"  Error reduction: {error_reduction_1:.4f}")
    
    # Run iteration 2
    print("\n[5/5] Running HCCA calibration (iteration 2)...")
    claims_iter2, report2 = batch_calibrate(
        list(claims_iter1),
        weights=DEFAULT_WEIGHTS,
        max_iterations=1,
        decay_factor=0.7,
    )
    
    errors_iter2 = []
    for claim in claims_iter2:
        if claim.id in actual_accuracies:
            error = abs(claim.confidence - actual_accuracies[claim.id])
            errors_iter2.append(error)
    
    mean_error_iter2 = sum(errors_iter2) / len(errors_iter2) if errors_iter2 else 0.5
    error_reduction_total = initial_mean_error - mean_error_iter2
    error_reduction_iter2 = mean_error_iter1 - mean_error_iter2
    
    print(f"  Mean error after iteration 2: {mean_error_iter2:.4f}")
    print(f"  Total error reduction: {error_reduction_total:.4f}")
    print(f"  Iteration 2 reduction: {error_reduction_iter2:.4f}")
    
    # Check verification criteria
    verification_pass = error_reduction_total > 0.3
    
    print(f"\n  Verification: calibration_error_reduction > 0.3")
    print(f"    Actual: {error_reduction_total:.4f}")
    print(f"    Status: {'PASS' if verification_pass else 'FAIL'}")
    
    # Build results
    results = {
        "experiment": "E2",
        "name": "confidence-calibration",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_claims": len(claims_list),
        "n_with_ground_truth": len([e for e in actual_accuracies.values()]),
        "calibration_formula": "C = 0.19×Local + 0.28×Direct + 0.30×Transitive + 0.23×Prior  # E11 optimized",
        "iterations": {
            "initial": {
                "mean_calibration_error": initial_mean_error,
                "n_errors": len(initial_errors),
            },
            "iteration_1": {
                "mean_calibration_error": mean_error_iter1,
                "error_reduction": error_reduction_1,
                "claims_changed": report1[0]["changed_claims"] if report1 else 0,
            },
            "iteration_2": {
                "mean_calibration_error": mean_error_iter2,
                "total_error_reduction": error_reduction_total,
                "iteration_2_reduction": error_reduction_iter2,
                "claims_changed": report2[0]["changed_claims"] if report2 else 0,
            },
        },
        "verification": {
            "criterion": "calibration_error_reduction > 0.3 after 2 iterations",
            "actual_reduction": error_reduction_total,
            "passed": verification_pass,
        },
        "weights_used": {
            "local": DEFAULT_WEIGHTS.local,
            "direct": DEFAULT_WEIGHTS.direct,
            "transitive": DEFAULT_WEIGHTS.transitive,
            "prior": DEFAULT_WEIGHTS.prior,
        },
        "sample_errors": {
            "initial": initial_errors[:10],
            "after_iter1": errors_iter1[:10],
            "after_iter2": errors_iter2[:10],
        },
    }
    
    return results


if __name__ == "__main__":
    results = run_confidence_calibration_experiment()
    
    # Save results
    output_path = "/home/aaron/projects/conjecture/research/rnd-sprint-2026-05-04/E2-results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print("\n" + "=" * 70)