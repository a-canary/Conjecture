import os
import sys
import yaml
import json
import subprocess
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Ensure src is in python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from processing.llm.llm_manager import LLMManager

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ConjectureEvaluator:
    def __init__(self, config_path, output_dir):
        # Load environment variables
        load_dotenv(project_root / ".env")

        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize LLM Judge
        self.llm_manager = LLMManager()
        logger.info(
            f"LLM Judge initialized with provider: {self.llm_manager.primary_provider}"
        )

        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Load Rubrics
        self.rubrics_dir = self.config_path.parent
        with open(self.rubrics_dir / "process_rubric.md", "r") as f:
            self.process_rubric = f.read()
        with open(self.rubrics_dir / "product_rubric.md", "r") as f:
            self.product_rubric = f.read()

    def run_all(self):
        results = []
        for use_case in self.config.get("use_cases", []):
            logger.info(f"Running evaluation for: {use_case['id']}")
            result = self.run_single_case(use_case)
            results.append(result)

        self.save_summary(results)

    def run_single_case(self, use_case):
        case_id = use_case["id"]
        prompt = use_case["prompt"]

        # 1. Execute Conjecture CLI
        start_time = datetime.now()

        # We use the simple CLI wrapper for testing
        cli_script = project_root / "demo" / "simple_conjecture_cli.py"

        logger.info(f"Executing prompt: {prompt}")
        try:
            # Run the CLI command
            # Using --prompt to pass the prompt directly if supported, or input via stdin
            # Assuming simple_conjecture_cli.py accepts arguments or we need to adapt
            # We'll try to pass it as an argument "run <prompt>" which is common
            cmd = [sys.executable, str(cli_script), "run", prompt]

            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout per case
            )

            logs = process.stderr + "\n" + process.stdout

            # Assume the CLI outputs the final artifact to a specific location or we parse it from stdout
            # For now, we'll treat stdout as the artifact if not found elsewhere
            artifact = process.stdout

            success = process.returncode == 0

        except subprocess.TimeoutExpired:
            logger.error(f"Timeout executing case {case_id}")
            logs = "TIMEOUT EXCEEDED"
            artifact = ""
            success = False
        except Exception as e:
            logger.error(f"Error executing case {case_id}: {e}")
            logs = str(e)
            artifact = ""
            success = False

        execution_time = (datetime.now() - start_time).total_seconds()

        # 2. Save Logs and Artifacts
        log_file = self.output_dir / f"{case_id}_log.txt"
        artifact_file = self.output_dir / f"{case_id}_artifact.txt"

        with open(log_file, "w", encoding="utf-8") as f:
            f.write(logs)

        with open(artifact_file, "w", encoding="utf-8") as f:
            f.write(artifact)

        # 3. Run LLM-as-Judge (Process)
        process_score = self.evaluate_process(use_case, logs)

        # 4. Run LLM-as-Judge (Product)
        product_score = self.evaluate_product(use_case, artifact)

        logger.info(
            f"Completed {case_id}. Process Score: {process_score.get('overall_score_average')}, Product Score: {product_score.get('overall_score_average')}"
        )

        return {
            "id": case_id,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": execution_time,
            "cli_success": success,
            "process_eval": process_score,
            "product_eval": product_score,
        }

    def evaluate_process(self, use_case, logs):
        prompt = f"""
        {self.process_rubric}
        
        === EVALUATION TARGET ===
        User Prompt: {use_case["prompt"]}
        
        System Execution Log:
        {logs[:20000]} # Truncate to avoid context limits if huge
        
        Provide your evaluation in JSON format as specified in the rubric.
        """

        try:
            # Using generic generate_response, assuming it returns an object with content or text
            response = self.llm_manager.generate_response(prompt)

            # Handle different response types from different providers
            if hasattr(response, "content"):
                text = response.content
            elif hasattr(response, "text"):
                text = response.text
            elif isinstance(response, str):
                text = response
            else:
                text = str(response)

            return self._parse_json_response(text)
        except Exception as e:
            logger.error(f"Process evaluation failed: {e}")
            return {"error": str(e), "overall_score_average": 0}

    def evaluate_product(self, use_case, artifact):
        prompt = f"""
        {self.product_rubric}
        
        === EVALUATION TARGET ===
        User Prompt: {use_case["prompt"]}
        Constraints: {use_case.get("expectations", [])}
        
        Final Artifact:
        {artifact[:20000]} # Truncate
        
        Provide your evaluation in JSON format as specified in the rubric.
        """

        try:
            response = self.llm_manager.generate_response(prompt)

            if hasattr(response, "content"):
                text = response.content
            elif hasattr(response, "text"):
                text = response.text
            elif isinstance(response, str):
                text = response
            else:
                text = str(response)

            return self._parse_json_response(text)
        except Exception as e:
            logger.error(f"Product evaluation failed: {e}")
            return {"error": str(e), "overall_score_average": 0}

    def _parse_json_response(self, text):
        try:
            # Clean markdown code blocks if present
            text = text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from LLM response: {text[:100]}...")
            return {"error": "JSON Parse Error", "raw_text": text[:500]}

    def save_summary(self, results):
        summary_path = self.output_dir / "evaluation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved evaluation summary to {summary_path}")


if __name__ == "__main__":
    evaluator = ConjectureEvaluator(
        config_path="tests/evaluation_config/use_cases.yaml",
        output_dir="tests/evaluation_results",
    )
    evaluator.run_all()
