#!/usr/bin/env python3
"""
XML Format Optimization - Comprehensive 4-Model Comparison Testing

Tests XML optimization effectiveness against baseline with statistical validation.
Measures claim format compliance improvement from 0% baseline to 60%+ target.
"""

import os
import sys
import json
import time
import asyncio
import re
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[OK] Environment loaded")
except ImportError:
    print("[FAIL] python-dotenv not available")

@dataclass
class XMLTestResult:
    """Enhanced test result for XML optimization testing"""
    model: str
    model_type: str
    approach: str  # "baseline" or "xml_optimized"
    test_case_id: str
    test_category: str
    prompt: str
    response: str
    response_time: float
    response_length: int
    status: str
    error: str = None
    
    # XML-specific metrics
    claims_generated: List[Dict[str, Any]] = None
    has_claim_format: bool = False
    claim_format_compliance: float = 0.0  # 0.0 to 1.0
    xml_claims_found: int = 0
    bracket_claims_found: int = 0
    structured_claims_found: int = 0
    
    # Quality metrics
    reasoning_steps: int = 0
    self_consistency_score: float = 0.0
    
    # Evaluation scores
    correctness_score: float = None
    completeness_score: float = None
    coherence_score: float = None
    reasoning_quality_score: float = None
    complexity_score: float = None
    
    timestamp: str = None
    
    def __post_init__(self):
        if self.claims_generated is None:
            self.claims_generated = []
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

# XML-optimized test cases focusing on claim generation
XML_TEST_CASES = [
    {
        "id": "scientific_concept",
        "category": "concept_exploration",
        "difficulty": "medium",
        "question": "Explain the concept of quantum entanglement and its implications for secure communication.",
        "expected_claims": 5,
        "claim_types": ["fact", "concept", "example", "hypothesis"],
        "focus": "scientific_accuracy and clear explanations"
    },
    {
        "id": "technology_analysis",
        "category": "technology_assessment",
        "difficulty": "medium",
        "question": "Evaluate the potential impact of artificial intelligence on job markets over the next decade.",
        "expected_claims": 6,
        "claim_types": ["fact", "concept", "example", "goal", "hypothesis"],
        "focus": "balanced analysis with evidence"
    },
    {
        "id": "historical_analysis",
        "category": "historical_reasoning",
        "difficulty": "hard",
        "question": "Analyze the primary factors that led to the Industrial Revolution and its long-term societal impacts.",
        "expected_claims": 7,
        "claim_types": ["fact", "concept", "example", "reference"],
        "focus": "causal reasoning and historical evidence"
    },
    {
        "id": "ethical_reasoning",
        "category": "ethical_analysis",
        "difficulty": "hard",
        "question": "Evaluate the ethical implications of gene editing technologies in human embryos.",
        "expected_claims": 6,
        "claim_types": ["fact", "concept", "hypothesis", "reference"],
        "focus": "ethical framework application"
    },
    {
        "id": "mathematical_concept",
        "category": "mathematical_reasoning",
        "difficulty": "medium",
        "question": "Explain the importance of prime numbers in modern cryptography and computer security.",
        "expected_claims": 5,
        "claim_types": ["fact", "concept", "example"],
        "focus": "technical accuracy and practical applications"
    }
]

# Model configurations for 4-model comparison
MODEL_CONFIGS = [
    {
        "name": "ibm/granite-4-h-tiny",
        "type": "tiny",
        "provider": "lm_studio",
        "url": "http://localhost:1234",
        "api_key": "",
        "description": "Tiny LLM (~3B parameters)"
    },
    {
        "name": "glm-z1-9b-0414",
        "type": "medium",
        "provider": "lm_studio",
        "url": "http://localhost:1234",
        "api_key": "",
        "description": "Medium LLM (9B parameters)"
    },
    {
        "name": "qwen3-4b-thinking-2507",
        "type": "medium",
        "provider": "lm_studio",
        "url": "http://localhost:1234",
        "api_key": "",
        "description": "Qwen thinking model (4B parameters)"
    },
    {
        "name": "zai-org/GLM-4.6",
        "type": "sota",
        "provider": "chutes",
        "url": "https://llm.chutes.ai/v1",
        "api_key": os.getenv("CHUTES_API_KEY", ""),
        "description": "State-of-the-art model (benchmark)"
    }
]

def make_api_call(prompt: str, model_config: Dict[str, Any], max_tokens: int = 2500) -> Dict[str, Any]:
    """Make API call to either LM Studio or Chutes"""
    try:
        import requests

        provider = model_config["provider"]
        url = model_config["url"]
        api_key = model_config["api_key"]
        model_name = model_config["name"]

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        data = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.3
        }

        # Fix API endpoint format
        if "chutes.ai" in url and url.endswith("/v1"):
            endpoint = f"{url}/chat/completions"
        else:
            endpoint = f"{url}/v1/chat/completions"

        start_time = time.time()
        response = requests.post(endpoint, headers=headers, json=data, timeout=300)
        response.raise_for_status()
        end_time = time.time()

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        return {
            "content": content,
            "response_time": end_time - start_time,
            "status": "success",
            "response_length": len(content)
        }

    except Exception as e:
        return {
            "content": f"Error: {str(e)}",
            "response_time": 0,
            "status": "error",
            "response_length": 0,
            "error": str(e)
        }

def generate_baseline_prompt(test_case: Dict[str, Any]) -> str:
    """Generate baseline prompt (original bracket format)"""
    return f"""You are Conjecture, an AI reasoning system that breaks down complex topics into specific claims.

**Task:** {test_case['question']}

**Instructions:**
Generate {test_case['expected_claims']} specific claims about this topic using this exact format:
[c1 | claim content | / confidence_level]
[c2 | claim content | / confidence_level]
etc.

Use confidence levels between 0.1 and 0.9. Include different types of claims: {', '.join(test_case['claim_types'])}.

Focus on: {test_case['focus']}"""

def generate_xml_optimized_prompt(test_case: Dict[str, Any]) -> str:
    """Generate XML-optimized prompt"""
    return f"""You are Conjecture, an AI reasoning system that creates structured claims using XML format.

<research_task>
{test_case['question']}
</research_task>

<claim_requirements>
Generate exactly {test_case['expected_claims']} high-quality claims using this XML format:

<claims>
  <claim type="fact" confidence="0.9">Your factual claim here</claim>
  <claim type="concept" confidence="0.8">Your conceptual claim here</claim>
  <claim type="example" confidence="0.7">Your specific example here</claim>
  <claim type="hypothesis" confidence="0.6">Your reasonable hypothesis here</claim>
</claims>
</claim_requirements>

<guidance>
- Use claim types: {', '.join(test_case['claim_types'])}
- Confidence scores should be between 0.1 and 0.9
- Focus on: {test_case['focus']}
- Ensure each claim is specific and verifiable
- Use the exact XML structure shown above
</guidance>"""

def extract_xml_claims(response: str) -> List[Dict[str, Any]]:
    """Extract XML claims from response"""
    claims = []
    
    # XML patterns
    xml_patterns = [
        r'<claim\s+type="([^"]*)"\s+confidence="([^"]*)"[^>]*>(.*?)</claim>',
        r'<claim\s+confidence="([^"]*)"\s+type="([^"]*)"[^>]*>(.*?)</claim>',
        r'<claim[^>]*type="([^"]*)"[^>]*confidence="([^"]*)"[^>]*>(.*?)</claim>',
    ]
    
    claim_counter = 1
    for pattern in xml_patterns:
        matches = re.findall(pattern, response, re.MULTILINE | re.DOTALL)
        for match in matches:
            try:
                if len(match) == 3:
                    if match[0].replace('.', '', 1).isdigit():  # confidence first
                        confidence_str, claim_type, content = match
                    else:  # type first
                        claim_type, confidence_str, content = match
                    
                    confidence = float(confidence_str)
                    if 0.0 <= confidence <= 1.0:
                        # Clean content
                        content = re.sub(r'<[^>]+>', '', content.strip())
                        content = re.sub(r'\s+', ' ', content)
                        
                        claims.append({
                            "id": f"c{claim_counter:03d}",
                            "type": claim_type,
                            "content": content,
                            "confidence": confidence,
                            "format": "xml"
                        })
                        claim_counter += 1
            except (ValueError, IndexError):
                continue
    
    return claims

def extract_bracket_claims(response: str) -> List[Dict[str, Any]]:
    """Extract bracket claims from response"""
    claims = []
    
    # Bracket patterns
    bracket_patterns = [
        r'\[c(\d+)\s*\|\s*([^|]+?)\s*\|\s*/\s*([0-9.]+)\s*\]',
        r'\[c(\d+)\|([^|]+?)\|/([0-9.]+)\]',
    ]
    
    for pattern in bracket_patterns:
        matches = re.findall(pattern, response, re.MULTILINE | re.DOTALL)
        for match in matches:
            try:
                claim_id, content, confidence_str = match
                confidence = float(confidence_str)
                if 0.0 <= confidence <= 1.0:
                    claims.append({
                        "id": f"c{claim_id}",
                        "type": "unknown",
                        "content": content.strip(),
                        "confidence": confidence,
                        "format": "bracket"
                    })
            except (ValueError, IndexError):
                continue
    
    return claims

def calculate_claim_format_compliance(result: XMLTestResult) -> float:
    """Calculate claim format compliance score (0.0 to 1.0)"""
    if result.status != "success":
        return 0.0
    
    total_claims = len(result.claims_generated)
    if total_claims == 0:
        return 0.0
    
    # Check if claims match expected format for approach
    if result.approach == "xml_optimized":
        xml_claims = sum(1 for c in result.claims_generated if c.get("format") == "xml")
        return xml_claims / total_claims
    else:  # baseline
        bracket_claims = sum(1 for c in result.claims_generated if c.get("format") == "bracket")
        return bracket_claims / total_claims

def analyze_reasoning_steps(response: str) -> int:
    """Count reasoning steps in response"""
    step_indicators = [
        r'\d+\.',  # Numbered steps
        r'•',      # Bullet points
        r'First,|Second,|Third,|Then,|Next,|Finally,',
        r'Because|Therefore|However|Thus|Hence',
        r'step|approach|method|analysis|evaluation'
    ]
    
    steps = 0
    for pattern in step_indicators:
        steps += len(re.findall(pattern, response, re.IGNORECASE))
    
    return min(steps, 15)  # Cap for normalization

def calculate_complexity_score(response: str) -> float:
    """Calculate response complexity score"""
    if not response:
        return 0.0
    
    # Factors: length, sentence complexity, technical terms
    word_count = len(response.split())
    sentence_count = len(re.split(r'[.!?]+', response))
    avg_sentence_length = word_count / max(sentence_count, 1)
    
    # Technical indicators
    technical_terms = len(re.findall(r'\b(analysis|evaluation|hypothesis|methodology|framework|algorithm|implementation)\b', response, re.IGNORECASE))
    
    # Normalize to 0-1 scale
    complexity = min((avg_sentence_length / 20.0) + (technical_terms / 10.0), 1.0)
    return complexity

def run_xml_test_for_model(model_config: Dict[str, Any], test_case: Dict[str, Any], approach: str) -> XMLTestResult:
    """Run a single XML optimization test"""
    print(f"    Testing {approach} approach...")
    
    if approach == "baseline":
        prompt = generate_baseline_prompt(test_case)
    else:
        prompt = generate_xml_optimized_prompt(test_case)
    
    result = make_api_call(prompt, model_config, max_tokens=2500)
    
    # Analyze response
    if result["status"] == "success":
        response = result["content"]
        
        # Extract claims based on approach
        if approach == "xml_optimized":
            claims = extract_xml_claims(response)
        else:
            claims = extract_bracket_claims(response)
        
        # Also try alternative extraction
        xml_claims = extract_xml_claims(response)
        bracket_claims = extract_bracket_claims(response)
        
        reasoning_steps = analyze_reasoning_steps(response)
        complexity = calculate_complexity_score(response)
        
        test_result = XMLTestResult(
            model=model_config["name"],
            model_type=model_config["type"],
            approach=approach,
            test_case_id=test_case["id"],
            test_category=test_case["category"],
            prompt=prompt,
            response=response,
            response_time=result["response_time"],
            response_length=result["response_length"],
            status=result["status"],
            claims_generated=claims,
            has_claim_format=len(claims) > 0,
            xml_claims_found=len(xml_claims),
            bracket_claims_found=len(bracket_claims),
            reasoning_steps=reasoning_steps,
            complexity_score=complexity
        )
        
        # Calculate compliance
        test_result.claim_format_compliance = calculate_claim_format_compliance(test_result)
        
        return test_result
    else:
        return XMLTestResult(
            model=model_config["name"],
            model_type=model_config["type"],
            approach=approach,
            test_case_id=test_case["id"],
            test_category=test_case["category"],
            prompt=prompt,
            response=result["content"],
            response_time=result["response_time"],
            response_length=result["response_length"],
            status=result["status"],
            error=result.get("error")
        )

async def run_comprehensive_xml_optimization_test():
    """Run comprehensive XML optimization test with 4-model comparison"""
    print("=" * 80)
    print("XML FORMAT OPTIMIZATION - COMPREHENSIVE 4-MODEL COMPARISON")
    print("Testing claim format compliance improvement from 0% baseline to 60%+ target")
    print("=" * 80)
    
    # Filter available models
    available_models = []
    for model in MODEL_CONFIGS:
        if model["provider"] == "chutes" and not model["api_key"]:
            print(f"[SKIP] {model['name']} - No API key")
            continue
        available_models.append(model)
    
    print(f"\nAvailable models: {len(available_models)}")
    for model in available_models:
        print(f"  - {model['name']} ({model['type']})")
    
    approaches = ["baseline", "xml_optimized"]
    total_tests = len(available_models) * len(approaches) * len(XML_TEST_CASES)
    print(f"\nTotal tests to run: {total_tests}")
    print(f"Test cases: {len(XML_TEST_CASES)}")
    print(f"Approaches: {', '.join(approaches)}")
    
    all_results = []
    current_test = 0
    
    # Run model-by-model to prevent LM Studio reloading
    for model in available_models:
        print(f"\n{'=' * 80}")
        print(f"TESTING MODEL: {model['name']} ({model['type']})")
        print(f"{'=' * 80}")
        
        for approach in approaches:
            print(f"\n[Approach: {approach.upper()}]")
            
            for test_case in XML_TEST_CASES:
                current_test += 1
                print(f"\n[{current_test}/{total_tests}] {test_case['id']} ({test_case['category']})")
                
                try:
                    result = run_xml_test_for_model(model, test_case, approach)
                    all_results.append(result)
                    
                    if result.status == "success":
                        print(f"  [OK] {result.response_time:.1f}s | {result.response_length} chars | "
                              f"{len(result.claims_generated)} claims | {result.claim_format_compliance:.1%} compliance")
                    else:
                        print(f"  [FAIL] {result.error}")
                    
                    # Brief pause between requests
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    print(f"  [ERROR] {e}")
                    error_result = XMLTestResult(
                        model=model["name"],
                        model_type=model["type"],
                        approach=approach,
                        test_case_id=test_case["id"],
                        test_category=test_case["category"],
                        prompt="",
                        response=f"Error: {str(e)}",
                        response_time=0,
                        response_length=0,
                        status="error",
                        error=str(e)
                    )
                    all_results.append(error_result)
    
    # Save results
    print(f"\n{'=' * 80}")
    print("SAVING RESULTS")
    print(f"{'=' * 80}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("research/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / f"xml_optimization_comprehensive_{timestamp}.json"
    
    # Convert to serializable format
    results_data = {
        "experiment_id": f"xml_optimization_comprehensive_{timestamp}",
        "timestamp": datetime.now().isoformat(),
        "models_tested": [m["name"] for m in available_models],
        "approaches_tested": approaches,
        "test_cases": XML_TEST_CASES,
        "results": [asdict(r) for r in all_results]
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"[OK] Results saved to: {results_file}")
    
    # Generate analysis
    await generate_xml_optimization_analysis(all_results, results_file)
    
    return all_results

async def generate_xml_optimization_analysis(results: List[XMLTestResult], results_file: Path):
    """Generate comprehensive analysis of XML optimization results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_file = results_file.parent / f"xml_optimization_analysis_{timestamp}.md"
    
    # Separate successful results
    successful = [r for r in results if r.status == "success"]
    failed = [r for r in results if r.status != "success"]
    
    # Group by model and approach
    model_approach_results = defaultdict(list)
    for result in successful:
        key = f"{result.model} | {result.approach}"
        model_approach_results[key].append(result)
    
    # Calculate key metrics
    baseline_results = [r for r in successful if r.approach == "baseline"]
    xml_results = [r for r in successful if r.approach == "xml_optimized"]
    
    baseline_compliance = statistics.mean([r.claim_format_compliance for r in baseline_results]) if baseline_results else 0.0
    xml_compliance = statistics.mean([r.claim_format_compliance for r in xml_results]) if xml_results else 0.0
    
    compliance_improvement = xml_compliance - baseline_compliance
    compliance_improvement_pct = (compliance_improvement / max(baseline_compliance, 0.01)) * 100
    
    # Statistical significance test
    significance_result = calculate_statistical_significance(baseline_results, xml_results)
    
    # Generate report
    report = f"""# XML Format Optimization - Comprehensive Analysis Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Experiment:** XML Format Optimization with 4-Model Comparison

## Executive Summary

### 🎯 Hypothesis Test Results
**Hypothesis:** XML-based prompts will increase claim format compliance from 0% baseline to 60%+

**Results:**
- **Baseline Compliance:** {baseline_compliance:.1%}
- **XML Optimized Compliance:** {xml_compliance:.1%}
- **Improvement:** {compliance_improvement:+.1%} ({compliance_improvement_pct:+.1f}% relative)
- **Target Achievement:** {"✅ ACHIEVED" if xml_compliance >= 0.60 else "❌ NOT ACHIEVED"}
- **Statistical Significance:** {significance_result}

### 📊 Test Statistics
- **Total Tests:** {len(results)}
- **Successful Tests:** {len(successful)}
- **Failed Tests:** {len(failed)}
- **Success Rate:** {len(successful)/len(results):.1%}

## Detailed Results

### Model-by-Model Performance

"""
    
    for key, result_list in model_approach_results.items():
        if not result_list:
            continue
        
        model_name = key.split(" | ")[0]
        approach = key.split(" | ")[1]
        
        avg_compliance = statistics.mean([r.claim_format_compliance for r in result_list])
        avg_claims = statistics.mean([len(r.claims_generated) for r in result_list])
        avg_time = statistics.mean([r.response_time for r in result_list])
        avg_complexity = statistics.mean([r.complexity_score or 0 for r in result_list])
        
        report += f"""#### {key}
- **Tests Completed:** {len(result_list)}
- **Claim Format Compliance:** {avg_compliance:.1%}
- **Average Claims Generated:** {avg_claims:.1f}
- **Average Response Time:** {avg_time:.1f}s
- **Average Complexity Score:** {avg_complexity:.3f}

"""
    
    report += f"""## Key Findings

### 1. Claim Format Compliance Analysis
- **Baseline Performance:** {baseline_compliance:.1%} compliance with bracket format
- **XML Optimization Performance:** {xml_compliance:.1%} compliance with XML format
- **Improvement Magnitude:** {compliance_improvement:+.1%} absolute improvement
- **Target vs Actual:** {'✅ EXCEEDED 60% target' if xml_compliance >= 0.60 else f'❌ Below 60% target by {(0.60 - xml_compliance):.1%}'}

### 2. Model-Specific Performance
[Detailed analysis of how different model types responded to XML optimization]

### 3. Complexity Impact Analysis
- **Baseline Complexity:** {statistics.mean([r.complexity_score or 0 for r in baseline_results]):.3f}
- **XML Complexity:** {statistics.mean([r.complexity_score or 0 for r in xml_results]):.3f}
- **Complexity Change:** {statistics.mean([r.complexity_score or 0 for r in xml_results]) - statistics.mean([r.complexity_score or 0 for r in baseline_results]):+.3f}
- **Complexity Impact:** {'✅ Within +10% target' if abs(statistics.mean([r.complexity_score or 0 for r in xml_results]) - statistics.mean([r.complexity_score or 0 for r in baseline_results])) <= 0.1 else '❌ Exceeds +10% target'}

### 4. Statistical Significance
{significance_result}

## Recommendations

### Based on Results:
"""
    
    if xml_compliance >= 0.60:
        report += """✅ **DEPLOY XML OPTIMIZATION**
- XML optimization successfully achieved target compliance
- Recommend immediate deployment to production
- Monitor performance in real-world usage
"""
    else:
        report += """❌ **REFINE XML OPTIMIZATION**
- Current implementation does not meet 60% compliance target
- Recommend further refinement before deployment
- Consider hybrid approach or simplified XML schema
"""
    
    if abs(statistics.mean([r.complexity_score or 0 for r in xml_results]) - statistics.mean([r.complexity_score or 0 for r in baseline_results])) <= 0.1:
        report += """✅ **COMPLEXITY IMPACT ACCEPTABLE**
- XML optimization maintains reasonable complexity
- No significant performance degradation expected
"""
    else:
        report += """⚠️ **MONITOR COMPLEXITY IMPACT**
- XML optimization increases response complexity
- Consider optimization for production deployment
"""
    
    report += f"""
## Technical Details

### Test Configuration
- **Models Tested:** {len(set(r.model for r in successful))}
- **Test Cases:** {len(XML_TEST_CASES)}
- **Approaches Compared:** baseline vs xml_optimized
- **Statistical Threshold:** α=0.05

### Data Files
- **Raw Results:** `{results_file.name}`
- **Analysis Report:** `{analysis_file.name}`

---
*Report generated by XML Optimization Comprehensive Test Runner*
"""
    
    with open(analysis_file, 'w') as f:
        f.write(report)
    
    print(f"[OK] Analysis report saved to: {analysis_file}")
    
    # Print summary to console
    print(f"\n{'=' * 80}")
    print("XML OPTIMIZATION TEST SUMMARY")
    print(f"{'=' * 80}")
    print(f"Baseline Compliance: {baseline_compliance:.1%}")
    print(f"XML Compliance: {xml_compliance:.1%}")
    print(f"Improvement: {compliance_improvement:+.1%}")
    print(f"Target Achievement: {'✅ ACHIEVED' if xml_compliance >= 0.60 else '❌ NOT ACHIEVED'}")
    print(f"Statistical Significance: {significance_result}")
    print(f"{'=' * 80}")

def calculate_statistical_significance(baseline_results: List[XMLTestResult], xml_results: List[XMLTestResult]) -> str:
    """Calculate statistical significance of improvement"""
    if len(baseline_results) < 3 or len(xml_results) < 3:
        return "Insufficient data for statistical analysis"
    
    try:
        baseline_scores = [r.claim_format_compliance for r in baseline_results]
        xml_scores = [r.claim_format_compliance for r in xml_results]
        
        # Simple t-test approximation
        baseline_mean = statistics.mean(baseline_scores)
        xml_mean = statistics.mean(xml_scores)
        
        baseline_std = statistics.stdev(baseline_scores) if len(baseline_scores) > 1 else 0
        xml_std = statistics.stdev(xml_scores) if len(xml_scores) > 1 else 0
        
        # Pooled standard error
        n1, n2 = len(baseline_scores), len(xml_scores)
        pooled_se = ((baseline_std**2 / n1) + (xml_std**2 / n2)) ** 0.5
        
        if pooled_se == 0:
            return "Cannot calculate significance (zero variance)"
        
        # t-statistic
        t_stat = (xml_mean - baseline_mean) / pooled_se
        
        # Simple significance assessment
        if abs(t_stat) > 2.0:  # Approximate p < 0.05
            return f"✅ Statistically significant (t={t_stat:.2f}, p<0.05)"
        elif abs(t_stat) > 1.5:  # Approximate p < 0.1
            return f"⚠️ Marginally significant (t={t_stat:.2f}, p<0.1)"
        else:
            return f"❌ Not statistically significant (t={t_stat:.2f}, p>0.1)"
            
    except Exception as e:
        return f"Statistical analysis failed: {str(e)}"

if __name__ == "__main__":
    asyncio.run(run_comprehensive_xml_optimization_test())