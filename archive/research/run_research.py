#!/usr/bin/env python3
"""
Main Research Runner
Orchestrates all Conjecture research experiments
"""

import asyncio
import argparse
import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Load environment variables from .env files
try:
    from dotenv import load_dotenv
    # Try to load .env from project root and research directory
    project_root = Path(__file__).parent.parent
    research_dir = Path(__file__).parent

    # Load .env files in order of precedence
    for env_file in [research_dir / '.env', project_root / '.env']:
        if env_file.exists():
            load_dotenv(env_file)
            print(f"Loaded environment from: {env_file}")
except ImportError:
    print("Warning: python-dotenv not available, using system environment variables only")

from config.common import ProviderConfig
from experiments.experiment_framework import ExperimentFramework
from experiments.hypothesis_experiments import HypothesisExperiments
from experiments.model_comparison import ModelComparisonSuite
from experiments.llm_judge import LLMJudge
from experiments.baseline_comparison import BaselineComparisonSuite, BaselineType
from test_cases.test_case_generator import TestCaseGenerator
from analysis.experiment_analyzer import ExperimentAnalyzer

def substitute_env_vars(config_dict):
    """
    Recursively substitute environment variables in configuration values
    Supports ${VAR} and ${VAR:-default} syntax
    """
    if isinstance(config_dict, dict):
        return {k: substitute_env_vars(v) for k, v in config_dict.items()}
    elif isinstance(config_dict, list):
        return [substitute_env_vars(item) for item in config_dict]
    elif isinstance(config_dict, str):
        # Replace ${VAR:-default} patterns
        def replace_var(match):
            var_expr = match.group(1)
            if ':-' in var_expr:
                var_name, default_value = var_expr.split(':-', 1)
                return os.getenv(var_name, default_value)
            else:
                return os.getenv(var_expr, '')

        # Handle both ${VAR} and ${VAR:-default} patterns
        pattern = r'\$\{([^}]+)\}'
        result = re.sub(pattern, replace_var, config_dict)

        # Convert string boolean/numeric values to proper types
        if result.lower() == 'true':
            return True
        elif result.lower() == 'false':
            return False
        elif result.isdigit():
            return int(result)
        elif result.replace('.', '').isdigit():
            try:
                return float(result)
            except ValueError:
                pass

        return result
    else:
        return config_dict

class ResearchRunner:
    """Main orchestrator for Conjecture research"""

    def __init__(self, config_file: str = None):
        self.config_file = config_file or "research/config.json"
        self.config = self._load_config()
        self.framework = None
        self.judge = None
        self.baseline_suite = None
        self.analyzer = ExperimentAnalyzer()

    def _load_config(self) -> dict:
        """Load research configuration"""
        config_path = Path(self.config_file)

        if config_path.exists():
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            # Substitute environment variables
            return substitute_env_vars(config_data)
        else:
            # Default configuration with environment variable support
            default_config = {
                "providers": [
                    {
                        "url": "${OLLAMA_API_URL:-http://localhost:11434}",
                        "api_key": "${OLLAMA_API_KEY:-}",
                        "model": "${OLLAMA_MODEL:-llama2}"
                    },
                    {
                        "url": "${LM_STUDIO_API_URL:-http://localhost:1234}",
                        "api_key": "${LM_STUDIO_API_KEY:-}",
                        "model": "${LM_STUDIO_MODEL:-ibm/granite-4-h-tiny}"
                    },
                    {
                        "url": "${CHUTES_API_URL:-https://llm.chutes.ai/v1}",
                        "api_key": "${CHUTES_API_KEY:-}",
                        "model": "${CHUTES_MODEL:-zai-org/GLM-4.6-FP8}"
                    },
                    {
                        "url": "${OPENROUTER_API_URL:-https://openrouter.ai/api/v1}",
                        "api_key": "${OPENROUTER_API_KEY:-}",
                        "model": "${OPENROUTER_MODEL:-openai/gpt-3.5-turbo}"
                    }
                ],
                "judge_model": "${JUDGE_MODEL:-chutes:zai-org/GLM-4.6-FP8}",
                "experiments": {
                    "hypothesis_validation": "${HYPOTHESIS_VALIDATION:-true}",
                    "model_comparison": "${MODEL_COMPARISON:-true}",
                    "baseline_comparison": "${BASELINE_COMPARISON:-true}",
                    "generate_test_cases": "${GENERATE_TEST_CASES:-true}"
                },
                "output": {
                    "save_results": "${SAVE_RESULTS:-true}",
                    "generate_visualizations": "${GENERATE_VISUALIZATIONS:-true}",
                    "create_reports": "${CREATE_REPORTS:-true}"
                }
            }

            # Substitute environment variables before saving
            substituted_config = substitute_env_vars(default_config)
            
            # Save template config (with environment variables) for user reference
            config_path.parent.mkdir(parents=True, exist_ok=True)
            template_config_path = config_path.with_suffix('.template.json')
            with open(template_config_path, 'w') as f:
                json.dump(default_config, f, indent=2)

            # Save actual config for immediate use
            with open(config_path, 'w') as f:
                json.dump(substituted_config, f, indent=2)

            print(f"Created config template at {template_config_path}")
            print(f"Created active config at {config_path}")
            print("Configure your environment variables in .env files")

            return substituted_config

    async def initialize(self):
        """Initialize all components"""
        print("🔧 Initializing research framework...")

        # Setup providers
        providers = []
        for provider_config in self.config["providers"]:
            providers.append(ProviderConfig(
                url=provider_config["url"],
                api_key=provider_config["api_key"],
                model=provider_config["model"]
            ))

        # Initialize framework
        self.framework = ExperimentFramework()
        success = await self.framework.initialize(providers)

        if not success:
            print("❌ Failed to initialize experiment framework")
            return False

        # Initialize judge
        self.judge = LLMJudge(self.framework.llm_manager, self.config["judge_model"])

        # Initialize baseline comparison suite
        self.baseline_suite = BaselineComparisonSuite(self.framework, self.config["judge_model"])

        print("✅ Research framework initialized successfully")
        return True

    async def generate_test_cases(self, count: int = 5):
        """Generate test cases for experiments"""
        print(f"📝 Generating {count} test cases per type...")

        generator = TestCaseGenerator()
        generated_files = generator.generate_test_suite(count_per_type=count)

        print(f"✅ Generated {len(generated_files)} test case files")
        return generated_files

    async def run_hypothesis_validation(self):
        """Run hypothesis validation experiments"""
        if not str(self.config["experiments"]["hypothesis_validation"]).lower() in ['true', '1', 'yes']:
            print("⏭️  Skipping hypothesis validation experiments")
            return None

        print("🧪 Running hypothesis validation experiments...")

        hypothesis_experiments = HypothesisExperiments(self.framework)
        results = await hypothesis_experiments.run_all_hypothesis_tests()

        # Generate report
        report = await hypothesis_experiments.generate_hypothesis_report(results)

        if str(self.config["output"]["save_results"]).lower() in ['true', '1', 'yes']:
            report_path = Path("research/analysis/hypothesis_validation_report.md")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"✅ Hypothesis validation report saved to {report_path}")

        return results

    async def run_model_comparison(self):
        """Run model comparison experiments"""
        if not str(self.config["experiments"]["model_comparison"]).lower() in ['true', '1', 'yes']:
            print("⏭️  Skipping model comparison experiments")
            return None

        print("🔬 Running model comparison experiments...")

        comparison_suite = ModelComparisonSuite(self.framework)
        results = await comparison_suite.run_comprehensive_comparison()

        # Generate report
        report = await comparison_suite.generate_model_comparison_report(results)

        if str(self.config["output"]["save_results"]).lower() in ['true', '1', 'yes']:
            report_path = Path("research/analysis/model_comparison_report.md")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"✅ Model comparison report saved to {report_path}")

        return results

    async def run_custom_experiment(self, experiment_config_path: str):
        """Run a custom experiment from config file"""
        print(f"🎯 Running custom experiment from {experiment_config_path}...")

        try:
            with open(experiment_config_path, 'r') as f:
                experiment_config = json.load(f)

            # Convert to ExperimentConfig object
            from experiments.experiment_framework import ExperimentConfig, ExperimentType, EvaluationMetric

            config = ExperimentConfig(
                experiment_id=experiment_config["experiment_id"],
                experiment_type=ExperimentType(experiment_config["experiment_type"]),
                name=experiment_config["name"],
                description=experiment_config["description"],
                hypothesis=experiment_config["hypothesis"],
                models_to_test=experiment_config["models_to_test"],
                test_cases=experiment_config["test_cases"],
                metrics=[EvaluationMetric(m) for m in experiment_config["metrics"]],
                parameters=experiment_config["parameters"]
            )

            # Run experiment
            run = await self.framework.run_experiment(config)

            print(f"✅ Custom experiment completed: {run.run_id}")
            return run

        except Exception as e:
            print(f"❌ Failed to run custom experiment: {e}")
            return None

    async def analyze_results(self):
        """Analyze all experiment results"""
        print("📊 Analyzing experiment results...")

        # Load all results
        results = self.analyzer.load_experiment_results()

        if not results:
            print("⚠️  No experiment results found to analyze")
            return

        # Analyze each experiment
        summaries = []
        for result in results:
            summary = self.analyzer.save_analysis(result)
            summaries.append(summary)

        # Generate comprehensive report
        if str(self.config["output"]["create_reports"]).lower() in ['true', '1', 'yes']:
            report = self.analyzer.generate_comprehensive_report()
            report_path = Path("research/analysis/comprehensive_analysis_report.md")
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"✅ Comprehensive analysis report saved to {report_path}")

        return summaries

    async def run_baseline_comparison(self):
        """Run baseline comparison experiments"""
        if not str(self.config["experiments"]["baseline_comparison"]).lower() in ['true', '1', 'yes']:
            print("⏭️  Skipping baseline comparison experiments")
            return None

        print("🏁 Running baseline comparison experiments...")

        # Load test cases for comparison
        test_cases = self._load_comparison_test_cases()

        if not test_cases:
            print("⚠️  No test cases found for baseline comparison")
            return None

        # Define models to test
        models_to_test = []
        for provider in self.config["providers"]:
            if provider.get("model"):
                models_to_test.append(provider["model"])

        if not models_to_test:
            print("⚠️  No models configured for testing")
            return None

        # Define baseline types to compare against
        baseline_types = [
            BaselineType.DIRECT_PROMPT,
            BaselineType.FEW_SHOT,
            BaselineType.CHAIN_OF_THOUGHT,
            BaselineType.ZERO_SHOT_COT
        ]

        # Run A/B tests
        comparison_results = await self.baseline_suite.run_ab_test(
            experiment_config=None,  # Not using formal experiment config for direct comparison
            baseline_types=baseline_types,
            test_cases=test_cases,
            models_to_test=models_to_test
        )

        # Generate comparative report
        report = await self.baseline_suite.generate_comparative_report(comparison_results)

        if str(self.config["output"]["save_results"]).lower() in ['true', '1', 'yes']:
            # Save results
            results_path = Path("research/results/baseline_comparison_results.json")
            results_path.parent.mkdir(parents=True, exist_ok=True)
            self.baseline_suite.save_comparison_results(str(results_path))
            print(f"✅ Baseline comparison results saved to {results_path}")

            # Save report
            report_path = Path("research/analysis/baseline_comparison_report.md")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"✅ Baseline comparison report saved to {report_path}")

        return comparison_results

    def _load_comparison_test_cases(self) -> List[Dict[str, Any]]:
        """Load test cases for baseline comparison"""
        test_cases = []

        # Try to load from file first
        test_cases_file = Path("research/test_cases/baseline_test_cases.json")
        if test_cases_file.exists():
            try:
                with open(test_cases_file, 'r') as f:
                    test_cases = json.load(f)
                print(f"📁 Loaded {len(test_cases)} test cases from {test_cases_file}")
                return test_cases
            except Exception as e:
                print(f"⚠️  Failed to load test cases from file: {e}")

        # Generate some default test cases for comparison
        print("📝 Generating default test cases for baseline comparison...")
        test_cases = [
            {
                "id": "reasoning_1",
                "question": "If a car travels 60 miles in 1 hour, and then travels 30 miles in 30 minutes, what is its average speed for the entire trip?",
                "context": "Average speed is calculated as total distance divided by total time.",
                "ground_truth": "The car travels 60 + 30 = 90 miles total. Total time is 1 hour + 0.5 hours = 1.5 hours. Average speed = 90 miles / 1.5 hours = 60 mph.",
                "examples": []
            },
            {
                "id": "analysis_1",
                "question": "Analyze the potential economic impacts of widespread remote work adoption.",
                "context": "",
                "ground_truth": "Economic impacts include reduced commercial real estate demand, changes in urban planning, shift in consumer spending patterns, potential productivity changes, and redistribution of economic activity from city centers to suburbs.",
                "examples": []
            },
            {
                "id": "creative_1",
                "question": "Write a short story about a robot that discovers emotions for the first time.",
                "context": "",
                "ground_truth": "Should be a creative story with robot protagonist experiencing emotions, showing character development and emotional depth.",
                "examples": []
            },
            {
                "id": "factual_1",
                "question": "What are the main differences between renewable and non-renewable energy sources?",
                "context": "",
                "ground_truth": "Renewable sources (solar, wind, hydro, geothermal) replenish naturally and have lower environmental impact, while non-renewable sources (coal, oil, natural gas) are finite and cause more pollution. Renewable sources often have intermittent availability, while non-renewable provide consistent power.",
                "examples": []
            },
            {
                "id": "problem_solving_1",
                "question": "How would you optimize a delivery route for 10 locations in a city to minimize total travel distance?",
                "context": "This is a variation of the traveling salesman problem.",
                "ground_truth": "Use algorithms like nearest neighbor, genetic algorithms, or optimization solvers. Consider factors like traffic patterns, delivery windows, vehicle capacity, and route constraints.",
                "examples": []
            }
        ]

        # Save the generated test cases for future use
        test_cases_file.parent.mkdir(parents=True, exist_ok=True)
        with open(test_cases_file, 'w') as f:
            json.dump(test_cases, f, indent=2)
        print(f"💾 Saved {len(test_cases)} test cases to {test_cases_file}")

        return test_cases

    async def run_full_research_suite(self):
        """Run the complete research suite"""
        print("🚀 Starting full Conjecture research suite...")
        print("=" * 60)

        start_time = datetime.now()

        try:
            # Initialize
            if not await self.initialize():
                return

            # Generate test cases
            if str(self.config["experiments"]["generate_test_cases"]).lower() in ['true', '1', 'yes']:
                await self.generate_test_cases(count=3)

            # Run hypothesis validation
            hypothesis_results = await self.run_hypothesis_validation()

            # Run model comparison
            comparison_results = await self.run_model_comparison()

            # Run baseline comparison
            baseline_results = await self.run_baseline_comparison()

            # Analyze results
            summaries = await self.analyze_results()

            # Final summary
            end_time = datetime.now()
            duration = end_time - start_time

            print("\n" + "=" * 60)
            print("🎉 RESEARCH SUITE COMPLETED")
            print("=" * 60)
            print(f"Duration: {duration}")
            print(f"Experiments run: {len(summaries) if summaries else 0}")

            if summaries:
                successful = [s for s in summaries if s.success_rate > 0.8]
                hypotheses_supported = [s for s in summaries if s.hypothesis_supported]

                print(f"Successful experiments: {len(successful)}/{len(summaries)}")
                print(f"Hypotheses supported: {len(hypotheses_supported)}/{len(summaries)}")

                if hypotheses_supported:
                    print("\n🏆 Key Validated Hypotheses:")
                    for summary in hypotheses_supported:
                        print(f"  ✅ {summary.experiment_name}")

            print(f"\n📁 Results saved to: research/results/")
            print(f"📈 Analysis saved to: research/analysis/")

        except Exception as e:
            print(f"❌ Research suite failed: {e}")
            import traceback
            traceback.print_exc()

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Conjecture Research Runner")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--generate-tests", action="store_true",
                       help="Only generate test cases")
    parser.add_argument("--hypothesis", action="store_true",
                       help="Only run hypothesis validation experiments")
    parser.add_argument("--comparison", action="store_true",
                       help="Only run model comparison experiments")
    parser.add_argument("--baseline", action="store_true",
                       help="Only run baseline comparison experiments")
    parser.add_argument("--analyze", action="store_true",
                       help="Only analyze existing results")
    parser.add_argument("--custom", help="Run custom experiment from config file")
    parser.add_argument("--full", action="store_true",
                       help="Run full research suite (default)")

    args = parser.parse_args()

    # Default to full suite if no specific action requested
    if not any([args.generate_tests, args.hypothesis, args.comparison,
                args.baseline, args.analyze, args.custom]):
        args.full = True

    runner = ResearchRunner(args.config)

    if args.generate_tests:
        await runner.initialize()
        await runner.generate_test_cases()
    elif args.hypothesis:
        await runner.initialize()
        await runner.run_hypothesis_validation()
    elif args.comparison:
        await runner.initialize()
        await runner.run_model_comparison()
    elif args.baseline:
        await runner.initialize()
        await runner.run_baseline_comparison()
    elif args.analyze:
        await runner.analyze_results()
    elif args.custom:
        await runner.initialize()
        await runner.run_custom_experiment(args.custom)
    elif args.full:
        await runner.run_full_research_suite()

if __name__ == "__main__":
    asyncio.run(main())