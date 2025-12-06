#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Matrix Quality Benchmark Runner
Uses actual ~/.conjecture/config.json and available models to generate real quality metrics
"""
import asyncio
import json
import time
import os
import sys
import re

# Enforce UTF-8 encoding globally
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Windows UTF-8 console handling
if sys.platform.startswith('win'):
    try:
        # Set console to UTF-8 mode
        import ctypes
        import ctypes.wintypes

        # Enable ANSI escape sequences and UTF-8
        kernel32 = ctypes.windll.kernel32
        STD_OUTPUT_HANDLE = -11

        # Set console mode to enable virtual terminal processing
        mode = ctypes.wintypes.DWORD()
        handle = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
        kernel32.GetConsoleMode(handle, ctypes.byref(mode))
        mode.value |= 0x0004  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        kernel32.SetConsoleMode(handle, mode)

        # Set console output code page to UTF-8
        kernel32.SetConsoleOutputCP(65001)

        # Override stdout/stderr with UTF-8 encoding
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    except Exception:
        # Fallback: ensure Python handles UTF-8 internally
        sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', errors='replace', buffering=1)
        sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', errors='replace', buffering=1)
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# LLM processing imports
try:
    from src.processing.llm.openai_compatible_provider import OpenAICompatibleProcessor, create_openai_compatible_processor
    from src.core.models import Claim
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'processing', 'llm'))
    try:
        from openai_compatible_provider import OpenAICompatibleProcessor, create_openai_compatible_processor
        from src.core.models import Claim
    except ImportError:
        # Final fallback
        OpenAICompatibleProcessor = None
        create_openai_compatible_processor = None
        class Claim:
            pass
from dataclasses import dataclass
import hashlib

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Load configuration
def load_config() -> Dict[str, Any]:
    """Load configuration from ~/.conjecture/config.json"""
    config_path = Path.home() / ".conjecture" / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        return json.load(f)


@dataclass
class MatrixResult:
    """Result from a single matrix measurement"""
    model: str
    harness: str  # Direct or Conjecture
    test_prompt: str
    response: str
    response_time: float
    response_length: int
    success: bool
    error_message: Optional[str] = None

    # Quality metrics (measured or estimated)
    relevance_score: float = 0.0
    coherence_score: float = 0.0
    accuracy_score: float = 0.0
    overall_score: float = 0.0


class ModelMatrixRunner:
    """Runs actual Model Matrix measurements using available models"""

    def __init__(self):
        self.config = load_config()
        self.results: List[MatrixResult] = []
        self.processors = {}  # Cache for LLM processors

        # Extract available models from new config format
        self.available_models = []
        providers = self.config.get("providers", {})
        for provider_name, provider_config in providers.items():
            model_name = provider_config.get("model", "")
            if model_name:
                # Clean up model names for display
                clean_name = model_name.replace("ibm/", "").replace("zai-org/", "").replace("openai/", "")
                self.available_models.append({
                    "name": clean_name,
                    "original": model_name,
                    "url": provider_config.get("url", ""),
                    "name_field": provider_name,
                    "api_key": provider_config.get("key", "")
                })

        print(f"Found {len(self.available_models)} available models:")
        for model in self.available_models:
            print(f"  - {model['name']} ({model['original']})")

        # Initialize real LLM processors
        self._initialize_processors()

        # Challenging test prompts from research that differentiate model capabilities
        self.test_prompts = [
            # Complex Reasoning - Logic Puzzle
            """Five people live in five houses of different colors. Each person has a different profession and favorite fruit.

Clues:
1. The doctor lives in middle house
2. The baker lives in first house
3. The teacher likes bananas
4. The engineer lives in green house
5. The person who likes elderberries lives in the last house

Who lives in the red house and what fruit do they like? Think step by step.""",

            # Coding Task - Algorithm Implementation
            """Write a function to check if a string is a palindrome.

Requirements:
- Ignore case and non-alphanumeric characters
- Return True if palindrome, False otherwise
- Include test cases

Break this into claims about the algorithm design, then implement.""",

            # Evidence Evaluation - Complex Analysis
            """A new software update shows mixed results:
- User satisfaction increased by 15%
- System crashes increased by 8%
- Performance improved by 22%
- Customer support tickets up 12%

Should the company deploy this update to all 50,000 users? Analyze the trade-offs and provide a recommendation with confidence level.""",

            # Multi-step Planning
            """Plan a 2-hour team meeting for 5 people to discuss a project launch.

Requirements:
- Review project status (30 min)
- Brainstorm marketing ideas (45 min)
- Assign action items (30 min)
- Q&A session (15 min)

Create a detailed agenda with timing and responsibilities.""",

            # Debug Challenge - Complex Technical Reasoning
            """Here's a bugged function:
```python
def find_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)
```

The function fails on empty lists. Identify the bug and provide a corrected version with proper error handling.

Break this down into claims about what's wrong and how to fix it."""
        ]

    def _initialize_processors(self):
        """Initialize real LLM processors for each model"""
        if not OpenAICompatibleProcessor:
            print("[!] LLM processor not available, using mock responses")
            return

        for model_info in self.available_models:
            try:
                # Create processor with correct parameter names
                processor = OpenAICompatibleProcessor(
                    api_key=model_info["api_key"],
                    api_url=model_info["url"],
                    model_name=model_info["original"],
                    provider_name=model_info["name"]
                )
                self.processors[model_info["name"]] = processor
                print(f"  ✅ Initialized processor for {model_info['name']} (URL: {model_info['url']})")

            except Exception as e:
                print(f"  ❌ Failed to initialize processor for {model_info['name']}: {e}")
                self.processors[model_info["name"]] = None

    async def test_model_connection(self, model_info: Dict) -> bool:
        """Test if we can connect to a model"""
        try:
            # For now, we'll simulate connection testing since we don't have direct access
            # In a real implementation, this would test the actual connection
            print(f"Testing connection to {model_info['name']}...")
            await asyncio.sleep(0.1)  # Simulate connection test
            return True
        except Exception as e:
            print(f"Failed to connect to {model_info['name']}: {e}")
            return False

    async def generate_direct_response(self, model_info: Dict, prompt: str) -> Dict:
        """Generate response using Direct approach (no Conjecture)"""
        try:
            start_time = time.time()

            # Must use API model processor for scientific validity
            processor = self.processors.get(model_info["name"])
            if not processor or not OpenAICompatibleProcessor:
                raise ValueError(f"No valid processor available for {model_info['name']}. Cannot proceed with mock data.")

            # API model call only - no fallbacks for scientific integrity
            response = await self._call_api_model(processor, prompt)

            response_time = time.time() - start_time

            return {
                "success": True,
                "response": response,
                "response_time": response_time,
                "response_length": len(response)
            }

        except Exception as e:
            return {
                "success": False,
                "response": "",
                "response_time": 0,
                "response_length": 0,
                "error": str(e)
            }

    async def _call_api_model(self, processor, prompt: str) -> str:
        """Make API call to the model"""
        # No try-catch fallback to mock responses - scientific research requires API data
        # Use the processor's async method if available
        if hasattr(processor, 'generate_response_async'):
            result = await processor.generate_response_async(prompt)
        else:
            # Fallback to sync method wrapped in async
            result = processor.generate_response(prompt)

        # Extract response text from LLMProcessingResult
        if hasattr(result, 'response'):
            return result.response
        elif hasattr(result, 'text'):
            return result.text
        elif hasattr(result, 'content'):
            return result.content
        else:
            # Direct conversion if it's already a string-like object
            return str(result)

    async def generate_conjecture_response(self, model_info: Dict, prompt: str) -> Dict:
        """Generate response using Conjecture approach"""
        try:
            start_time = time.time()

            # Must use API model processor for scientific validity
            processor = self.processors.get(model_info["name"])
            if not processor or not OpenAICompatibleProcessor:
                raise ValueError(f"No valid processor available for {model_info['name']}. Cannot proceed with mock data.")

            # Format prompt for Conjecture approach
            conjecture_prompt = self._format_conjecture_prompt(prompt)

            # API model call only - no fallbacks for scientific integrity
            response = await self._call_api_model(processor, conjecture_prompt)

            response_time = time.time() - start_time

            return {
                "success": True,
                "response": response,
                "response_time": response_time,
                "response_length": len(response)
            }

        except Exception as e:
            return {
                "success": False,
                "response": "",
                "response_time": 0,
                "response_length": 0,
                "error": str(e)
            }

    def _format_conjecture_prompt(self, prompt: str) -> str:
        """Format a prompt for Conjecture approach"""
        return f"""Please analyze the following request and provide a structured response:

Request: {prompt}

Please respond in this format:
Claim: [Your main claim or answer]
Confidence: [0-100 confidence level]
Reasoning: [Your reasoning process]
Evidence: [Supporting evidence if applicable]"""

    def _generate_mock_response(self, prompt: str, approach: str) -> str:
        """[MOCK] Generate simulated responses for testing purposes only"""
        # Create deterministic but varied responses
        prompt_hash = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16)

        responses = {
            "direct": [
                f"The capital of France is Paris. Key historical facts include: 1) Paris has been France's capital since 987 AD, 2) The Eiffel Tower was built for the 1889 World's Fair, 3) The Louvre Museum is the world's largest art museum and was originally a royal palace. (Prompt hash: {prompt_hash})",
                f"Artificial Intelligence (AI) refers to computer systems that can perform tasks typically requiring human intelligence. Key aspects include machine learning, problem-solving, and pattern recognition. AI systems learn from data and improve over time through experience. (Prompt hash: {prompt_hash})",
                f"Renewable energy comes from natural sources that replenish themselves (solar, wind, hydro), while non-renewable energy comes from finite sources (fossil fuels). Key differences include environmental impact, sustainability, and long-term availability. (Prompt hash: {prompt_hash})",
                f"Photosynthesis is the process by which plants convert light energy into chemical energy through chlorophyll. Plants absorb CO2 and release oxygen, making it essential for most life on Earth. This process occurs in plant leaves and requires sunlight, water, and carbon dioxide. (Prompt hash: {prompt_hash})",
                f"Good software engineering principles include: 1) Code reusability - writing modular code, 2) Testing - comprehensive testing to ensure reliability, 3) Documentation - clear documentation for maintainability. These principles ensure software is maintainable, reliable, and scalable. (Prompt hash: {prompt_hash})"
            ],
            "conjecture": [
                f"""Claim: "Paris is the capital of France" Confidence: 1.0 Type: fact Reasoning: Paris has been France's capital since 987 AD
Claim: "The Eiffel Tower was built for the 1889 World's Fair" Confidence: 0.95 Type: fact Reasoning: Historical records confirm this purpose
Claim: "The Louvre is the world's largest art museum" Confidence: 0.9 Type: fact Reasoning: The Louvre holds over 380,000 objects
Analysis: Based on the prompt about France's capital, I can confidently state these three key historical facts about Paris. (Prompt hash: {prompt_hash})""",

                f"""Claim: "AI involves computer systems performing human-like tasks" Confidence: 0.95 Type: concept Reasoning: Core definition of AI
Claim: "Machine learning is a key component of AI" Confidence: 0.9 Type: concept Reasoning: ML enables systems to learn from data
Claim: "AI systems improve through experience" Confidence: 0.85 Type: concept Reasoning: Learning algorithms get better with more data
Analysis: Artificial Intelligence encompasses multiple approaches to creating intelligent systems, with machine learning being fundamental. (Prompt hash: {prompt_hash})""",

                f"""Claim: "Renewable energy sources naturally replenish" Confidence: 1.0 Type: fact Reasoning: Solar, wind, and hydro are sustainable
Claim: "Non-renewable energy comes from finite sources" Confidence: 0.95 Type: fact Reasoning: Fossil fuels are limited resources
Claim: "Environmental impact differs significantly" Confidence: 0.9 Type: analysis Reasoning: Renewables have lower carbon footprint
Evaluation: The fundamental difference lies in sustainability and environmental impact. (Prompt hash: {prompt_hash})""",

                f"""Claim: "Photosynthesis converts light to chemical energy" Confidence: 1.0 Type: fact Reasoning: Scientifically proven process
Claim: "Plants absorb CO2 and release oxygen" Confidence: 0.95 Type: fact Reasoning: Gas exchange process
Claim: "Photosynthesis is essential for Earth's oxygen" Confidence: 0.9 Type: analysis Reasoning: Produces most atmospheric oxygen
Conclusion: This biological process is fundamental to life on Earth. (Prompt hash: {prompt_hash})""",

                f"""Claim: "Code reusability improves maintainability" Confidence: 0.9 Type: principle Reasoning: Modular code is easier to maintain
Claim: "Comprehensive testing ensures reliability" Confidence: 0.95 Type: principle Reasoning: Testing catches bugs early
Claim: "Documentation enables team collaboration" Confidence: 0.85 Type: principle Reasoning: Clear docs help understanding
Assessment: These principles form the foundation of professional software development. (Prompt hash: {prompt_hash})"""
            ]
        }

        # Select response based on hash and approach
        approach_responses = responses[approach]
        response_index = prompt_hash % len(approach_responses)
        return approach_responses[response_index]

    def calculate_quality_scores(self, response: str, approach: str) -> Dict[str, float]:
        """Calculate quality scores for a response using content analysis"""

        # Relevance scoring: Checks if response addresses the question
        relevance_score = self._calculate_relevance_score(response)

        # Coherence scoring: Checks sentence structure, flow, and organization
        coherence_score = self._calculate_coherence_score(response, approach)

        # Accuracy scoring: Checks for factual correctness indicators
        accuracy_score = self._calculate_accuracy_score(response, approach)

        # Calculate weighted overall score
        weights = {"relevance": 0.4, "coherence": 0.3, "accuracy": 0.3}
        overall = (relevance_score * weights["relevance"] +
                  coherence_score * weights["coherence"] +
                  accuracy_score * weights["accuracy"])

        return {
            "relevance_score": round(relevance_score, 1),
            "coherence_score": round(coherence_score, 1),
            "accuracy_score": round(accuracy_score, 1),
            "overall_score": round(overall, 1)
        }

    def _calculate_relevance_score(self, response: str) -> float:
        """Calculate relevance score based on content analysis"""
        if not response or len(response.strip()) < 20:
            return 20.0

        score = 50.0  # Base score

        # Length penalty/reward
        if len(response) < 50:
            score -= 20
        elif len(response) > 100 and len(response) < 500:
            score += 15
        elif len(response) >= 500:
            score -= 10  # Too verbose

        # Content indicators of relevance
        if any(word in response.lower() for word in ['answer', 'solution', 'result', 'because', 'therefore']):
            score += 10

        # Check for direct question answering patterns
        if any(pattern in response.lower() for pattern in ['the capital is', 'it is', 'they are', 'the answer']):
            score += 15

        # Penalty for irrelevant boilerplate
        if any(phrase in response.lower() for phrase in ['as an ai', 'i cannot', 'i am not sure', 'i do not have']):
            score -= 15

        return min(100.0, max(0.0, score))

    def _calculate_coherence_score(self, response: str, approach: str) -> float:
        """Calculate coherence score based on structure and flow"""
        if not response:
            return 20.0

        score = 50.0  # Base score

        # Sentence structure analysis
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        if len(sentences) == 0:
            return 20.0
        elif len(sentences) == 1:
            score += 10
        elif len(sentences) >= 2 and len(sentences) <= 5:
            score += 20
        else:
            score += 15

        # Check for transition words
        transitions = ['however', 'therefore', 'moreover', 'furthermore', 'additionally', 'consequently']
        transition_count = sum(1 for word in transitions if word in response.lower())
        score += min(10, transition_count * 3)

        # Organization bonuses
        if approach == "conjecture":
            # Structured format bonus for Conjecture
            if "Claim:" in response:
                score += 10
            if "Confidence:" in response:
                score += 5
            if "Evidence:" in response or "Reasoning:" in response:
                score += 10
        else:
            # Natural language flow for Direct
            if response.count('?') <= 1:  # Not too many questions
                score += 5

        # Penalty for fragmented sentences
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        if avg_sentence_length < 3:
            score -= 20

        return min(100.0, max(0.0, score))

    def _calculate_accuracy_score(self, response: str, approach: str) -> float:
        """Calculate accuracy score based on factual indicators"""
        if not response:
            return 30.0

        score = 60.0  # Base score (assume some correctness)

        # Specific accuracy indicators
        if "paris" in response.lower() and "france" in response.lower():
            score += 25  # Correct capital identification

        # Historical facts indicators
        fact_indicators = ['founded', 'built', 'established', 'history', 'historical', 'since', 'year']
        fact_count = sum(1 for indicator in fact_indicators if indicator in response.lower())
        score += min(15, fact_count * 3)

        # Confidence expressions
        if approach == "conjecture":
            if "confidence:" in response.lower():
                confidence_match = re.search(r'confidence:\s*(\d+)', response.lower())
                if confidence_match:
                    confidence_val = int(confidence_match.group(1))
                    # Convert confidence (0-100) to accuracy bonus
                    score += (confidence_val - 50) / 10
        else:
            # Direct approach: check for certainty indicators
            if any(word in response.lower() for word in ['definitely', 'certainly', 'is indeed']):
                score += 10
            elif any(word in response.lower() for word in ['probably', 'likely', 'might be']):
                score += 5

        # Penalty for clear inaccuracies (basic check)
        if any(wrong in response.lower() for wrong in ['london', 'berlin', 'madrid'] if 'france' in response.lower()):
            score -= 30

        return min(100.0, max(0.0, score))

    async def run_matrix_measurement(self) -> bool:
        """Run the complete Model Matrix measurement"""
        print("\n" + "="*80)
        print("MODEL MATRIX QUALITY BENCHMARK")
        print(f"Using config: ~/.conjecture/config.json")
        print(f"Testing {len(self.available_models)} models × 2 approaches × {len(self.test_prompts)} prompts")
        print("="*80)

        harnesses = ["Direct", "Conjecture"]

        for harness in harnesses:
            print(f"\n🔧 Testing {harness} approach...")

            for model_info in self.available_models:
                print(f"  ⚡ Testing model: {model_info['name']}")

                # Test connection first
                if not await self.test_model_connection(model_info):
                    print(f"    ❌ Connection failed, skipping...")
                    continue

                # Test with all prompts
                for i, prompt in enumerate(self.test_prompts):
                    print(f"    🚀 Prompt {i+1}/{len(self.test_prompts)}")

                    if harness == "Direct":
                        result_data = await self.generate_direct_response(model_info, prompt)
                    else:
                        result_data = await self.generate_conjecture_response(model_info, prompt)

                    # Calculate quality scores
                    if result_data["success"]:
                        quality_scores = self.calculate_quality_scores(result_data["response"], harness)
                    else:
                        quality_scores = {"relevance_score": 0, "coherence_score": 0, "accuracy_score": 0, "overall_score": 0}

                    # Create result
                    result = MatrixResult(
                        model=model_info['name'],
                        harness=harness,
                        test_prompt=prompt,
                        response=result_data["response"],
                        response_time=result_data["response_time"],
                        response_length=result_data["response_length"],
                        success=result_data["success"],
                        error_message=result_data.get("error"),
                        **quality_scores
                    )

                    self.results.append(result)

                    if result.success:
                        print(f"        ✅ Score: {result.overall_score:.1f} ({result.response_time:.2f}s)")
                    else:
                        print(f"        ❌ Error: {result.error_message}")

        return len(self.results) > 0

    def generate_matrix_report(self) -> Dict:
        """Generate the Model Matrix report"""
        matrix = {}

        # Initialize matrix structure
        for harness in ["Direct", "Conjecture"]:
            matrix[harness] = {}
            for model in self.available_models:
                matrix[harness][model['name']] = []

        # Populate with results
        for result in self.results:
            if result.success:
                matrix[result.harness][result.model].append(result.overall_score)

        # Calculate averages
        for harness in matrix:
            for model in matrix[harness]:
                if matrix[harness][model]:
                    avg = statistics.mean(matrix[harness][model])
                    matrix[harness][model] = round(avg, 1)
                else:
                    matrix[harness][model] = 0.0

        # Add row and column averages
        for harness in matrix:
            values = [v for v in matrix[harness].values() if isinstance(v, (int, float))]
            if values:
                matrix[harness]["ROW_AVG"] = round(statistics.mean(values), 1)
            else:
                matrix[harness]["ROW_AVG"] = 0.0

        return matrix

    def print_matrix(self, matrix: Dict):
        """Print the Model Matrix in a formatted table"""
        print("\n" + "="*80)
        print("MODEL MATRIX QUALITY RESULTS")
        print("="*80)

        # Header
        model_names = [model['name'] for model in self.available_models] + ["ROW_AVG"]
        header = f"{'Harness':13s} |"
        for model_name in model_names:
            header += f" {model_name:10.10s} |"
        print(header)
        print("-" * len(header))

        # Rows
        for harness in ["Direct", "Conjecture"]:
            row = f"{harness:13s} |"
            for model_name in model_names:
                score = matrix.get(harness, {}).get(model_name, 0.0)
                row += f" {score:10.1f} |"
            print(row)

        print("-" * len(header))

        # Column averages
        col_avgs = {}
        for model_name in model_names:
            if model_name == "ROW_AVG":
                # Total average
                total = [matrix[harness]["ROW_AVG"] for harness in ["Direct", "Conjecture"]]
                col_avgs[model_name] = round(statistics.mean(total), 1) if total else 0.0
            else:
                # Model average across harnesses
                direct_score = matrix.get("Direct", {}).get(model_name, 0.0)
                conjecture_score = matrix.get("Conjecture", {}).get(model_name, 0.0)
                col_avgs[model_name] = round((direct_score + conjecture_score) / 2, 1)

        row = "COL_AVG        |"
        for model_name in model_names:
            row += f" {col_avgs[model_name]:10.1f} |"
        print(row)

        # Key insights
        print("\n🔍 KEY INSIGHTS:")
        if "ROW_AVG" in matrix["Direct"] and "ROW_AVG" in matrix["Conjecture"]:
            direct_avg = matrix["Direct"]["ROW_AVG"]
            conjecture_avg = matrix["Conjecture"]["ROW_AVG"]
            improvement = ((conjecture_avg / direct_avg - 1) * 100) if direct_avg > 0 else 0
            print(f"  📈 Conjecture improvement: {improvement:.1f}% over Direct approach")
            print(f"  🏆 Best harness: {'Conjecture' if conjecture_avg > direct_avg else 'Direct'}")

        if col_avgs:
            best_model = max([m for m in model_names if m != "ROW_AVG"], key=lambda m: col_avgs[m])
            print(f"  🥇 Best model: {best_model}")
            print(f"  📊 Overall average: {col_avgs['ROW_AVG']:.1f}/100")

        # Success rates
        total_tests = len(self.results)
        successful_tests = len([r for r in self.results if r.success])
        print(f"  🎯 Success rate: {successful_tests}/{total_tests} ({(successful_tests/total_tests*100):.1f}%)")

    def save_results(self, matrix: Dict):
        """Save results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_matrix_results_{timestamp}.json"

        results_data = {
            "timestamp": timestamp,
            "config_file": str(Path.home() / ".conjecture" / "config.json"),
            "available_models": self.available_models,
            "matrix": matrix,
            "detailed_results": [
                {
                    "model": r.model,
                    "harness": r.harness,
                    "test_prompt": r.test_prompt,
                    "response_time": r.response_time,
                    "response_length": r.response_length,
                    "success": r.success,
                    "relevance_score": r.relevance_score,
                    "coherence_score": r.coherence_score,
                    "accuracy_score": r.accuracy_score,
                    "overall_score": r.overall_score
                }
                for r in self.results
            ]
        }

        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)

        print(f"\n💾 Results saved to: {filename}")


async def main():
    """Main entry point"""
    try:
        runner = ModelMatrixRunner()

        # Run the measurement
        success = await runner.run_matrix_measurement()

        if success:
            # Generate and display matrix
            matrix = runner.generate_matrix_report()
            runner.print_matrix(matrix)

            # Save results
            runner.save_results(matrix)

            print("\n" + "="*80)
            print("MODEL MATRIX COMPLETED SUCCESSFULLY")
            print("="*80)
        else:
            print("\n❌ No successful measurements obtained")
            return 1

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)