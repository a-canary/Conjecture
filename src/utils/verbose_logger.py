"""
Multi-Level Verbose Logging System for Conjecture
Distinguishes between validation (structural) and confidence assessment (quality)
"""

import time
from datetime import datetime
from typing import Optional
from enum import Enum

from .emoji_support import emoji_printer

class VerboseLevel(Enum):
    NONE = 0  # No verbose output
    USER = 1  # User tools + confidence assessments
    TOOLS = 2  # All tool calls with parameters
    DEBUG = 3  # Process logging + per-evaluation debugging

class VerboseLogger:
    def __init__(self, level: VerboseLevel = VerboseLevel.NONE):
        self.level = level
        self.start_time = None
        self.stats = {
            "claims_created": 0,
            "claims_assessed_confident": 0,
            "claims_needing_evaluation": 0,
            "tool_calls": 0,
            "support_relationships": 0,
            "evaluations": 0,
        }

    def _log(self, level: VerboseLevel, message: str, emoji_shortcode: str = ""):
        if self.level.value >= level.value:
            timestamp = (
                datetime.now().strftime("%H:%M:%S")
                if level == VerboseLevel.DEBUG
                else ""
            )
            prefix = f"[{timestamp}] " if timestamp else ""
            if emoji_shortcode:
                emoji_printer.print(f"{prefix}{emoji_shortcode} {message}")
            else:
                print(f"{prefix}{message}")

    # Level 1: User Communication + Confidence Assessments
    def user_tool_executed(self, tool_name: str, result: str):
        """Only tools that communicate with user (TellUser, AskUser)"""
        if tool_name in ["TellUser", "AskUser"]:
            self._log(VerboseLevel.USER, f"User message: {result}", ":speech_balloon:")

    def claim_assessed_confident(
        self, claim_id: str, confidence: float, threshold: float
    ):
        """Log when claim meets confidence threshold"""
        if confidence >= threshold:
            self.stats["claims_assessed_confident"] += 1
            self._log(
                VerboseLevel.USER,
                f"Claim confident: {claim_id} (confidence: {confidence:.2f} >= {threshold:.2f})",
                ":target:",
            )
        else:
            self.stats["claims_needing_evaluation"] += 1
            self._log(
                VerboseLevel.USER,
                f"Claim needs evaluation: {claim_id} (confidence: {confidence:.2f} < {threshold:.2f})",
                ":hourglass_flowing_sand:",
            )

    def claim_resolved(self, claim_id: str, confidence: float):
        """Log when claim is fully resolved"""
        self._log(
            VerboseLevel.USER,
            f"Claim resolved: {claim_id} (confidence: {confidence:.2f})",
            ":check_mark:",
        )

    def final_response(self, response: str):
        display_response = response[:200] + "..." if len(response) > 200 else response
        self._log(VerboseLevel.USER, f"Final response: {display_response}", ":target:")

    # Level 2: All Tool Calls with Parameters
    def tool_executed(self, tool_name: str, args: dict, result: dict = None):
        self.stats["tool_calls"] += 1

        if self.level.value >= VerboseLevel.TOOLS.value:
            # Format arguments for display
            if args:
                arg_strs = []
                for k, v in args.items():
                    if isinstance(v, str) and len(v) > 50:
                        v = v[:47] + "..."
                    arg_strs.append(f"{k}={v}")
                args_display = "(" + ", ".join(arg_strs) + ")"
            else:
                args_display = "()"

            self._log(
                VerboseLevel.TOOLS,
                f"Tool: {tool_name}{args_display}",
                ":hammer_and_wrench:",
            )

            # Show result summary if available
            if result and self.level.value >= VerboseLevel.DEBUG.value:
                success = result.get("success", False)
                result_type = "SUCCESS" if success else "ERROR"
                self._log(VerboseLevel.DEBUG, f"Result: {result_type}", "📋")

    def claim_validated(self, claim_id: str, validation_result: str):
        """Log structural validation results (not confidence)"""
        if self.level.value >= VerboseLevel.TOOLS.value:
            if validation_result == "valid":
                self._log(
                    VerboseLevel.TOOLS,
                    f"Claim validated: {claim_id} - structure OK",
                    "✅",
                )
            else:
                self._log(
                    VerboseLevel.TOOLS,
                    f"Claim validation failed: {claim_id} - {validation_result}",
                    "❌",
                )

    def claim_created(self, claim_id: str, content: str, confidence: float):
        self.stats["claims_created"] += 1
        if self.level.value >= VerboseLevel.TOOLS.value:
            content_display = content[:40] + "..." if len(content) > 40 else content
            self._log(
                VerboseLevel.TOOLS,
                f'Claim created: {claim_id} - "{content_display}" (confidence: {confidence:.2f})',
                "📝",
            )

    def support_added(self, supporter: str, supported: str):
        self.stats["support_relationships"] += 1
        if self.level.value >= VerboseLevel.TOOLS.value:
            self._log(
                VerboseLevel.TOOLS, f"Support added: {supporter} -> {supported}", "🔗"
            )

    # Level 3: Process Logging + Per-Evaluation Debugging
    def process_start(self, operation: str):
        if self.level.value >= VerboseLevel.DEBUG.value:
            self.start_time = time.time()
            self._log(VerboseLevel.DEBUG, f"Starting: {operation}", "🔍")

    def context_built(
        self, supporting: int, supported: int, semantic: int, tokens: int
    ):
        if self.level.value >= VerboseLevel.DEBUG.value:
            total = supporting + supported + semantic
            self._log(
                VerboseLevel.DEBUG,
                f"Context built: {total} claims ({supporting}↑, {supported}↓, {semantic}≈) {tokens} tokens",
                "🏗️",
            )

    def llm_processing(self, context_size: int):
        if self.level.value >= VerboseLevel.DEBUG.value:
            self._log(
                VerboseLevel.DEBUG, f"LLM processing: {context_size} characters", "🤖"
            )

    def evaluation_start(self, claim_id: str, evaluation_type: str):
        self.stats["evaluations"] += 1
        if self.level.value >= VerboseLevel.DEBUG.value:
            self._log(
                VerboseLevel.DEBUG,
                f"Evaluation started: {claim_id} ({evaluation_type})",
                "⚡",
            )

    def evaluation_complete(self, claim_id: str, result: str):
        if self.level.value >= VerboseLevel.DEBUG.value:
            self._log(
                VerboseLevel.DEBUG, f"Evaluation complete: {claim_id} -> {result}", "✨"
            )

    def claim_marked_dirty(self, claim_id: str, reason: str):
        if self.level.value >= VerboseLevel.DEBUG.value:
            self._log(
                VerboseLevel.DEBUG, f"Claim marked dirty: {claim_id} ({reason})", "🚩"
            )

    def tool_registry_loaded(self, core_count: int, optional_count: int):
        if self.level.value >= VerboseLevel.DEBUG.value:
            self._log(
                VerboseLevel.DEBUG,
                f"Tool registry loaded: {core_count} core, {optional_count} optional",
                "🛠️",
            )

    def confidence_threshold_updated(self, old_threshold: float, new_threshold: float):
        """Log when confidence threshold is changed"""
        if self.level.value >= VerboseLevel.USER.value:
            self._log(
                VerboseLevel.USER,
                f"Confidence threshold updated: {old_threshold:.2f} → {new_threshold:.2f}",
                "⚙️",
            )

    def error(self, message: str, error: Exception = None):
        """Always show errors regardless of verbose level"""
        error_detail = f": {str(error)}" if error else ""
        self._log(VerboseLevel.USER, f"Error: {message}{error_detail}", "❌")

    def finish(self):
        if self.level.value >= VerboseLevel.USER.value and self.start_time:
            duration = time.time() - self.start_time
            self._log(VerboseLevel.USER, f"Completed in {duration:.1f}s", "⏱️")

            if self.level.value >= VerboseLevel.TOOLS.value:
                stats_parts = []
                if self.stats["claims_created"] > 0:
                    stats_parts.append(f"{self.stats['claims_created']} claims")
                if self.stats["tool_calls"] > 0:
                    stats_parts.append(f"{self.stats['tool_calls']} tools")
                if self.stats["claims_assessed_confident"] > 0:
                    stats_parts.append(
                        f"{self.stats['claims_assessed_confident']} confident"
                    )
                if self.stats["claims_needing_evaluation"] > 0:
                    stats_parts.append(
                        f"{self.stats['claims_needing_evaluation']} need evaluation"
                    )
                if self.stats["evaluations"] > 0:
                    stats_parts.append(f"{self.stats['evaluations']} evaluations")

                if stats_parts:
                    self._log(
                        VerboseLevel.USER, f"Stats: {', '.join(stats_parts)}", "📊"
                    )

    def log_confidence_summary(self, claims):
        """Log summary of claim confidence assessments"""
        if self.level.value >= VerboseLevel.USER.value:
            confident_count = sum(1 for claim in claims if claim.is_confident)
            total_count = len(claims)
            if total_count > 0:
                confident_pct = (confident_count / total_count) * 100
                self._log(
                    VerboseLevel.USER,
                    f"Confidence summary: {confident_count}/{total_count} claims confident ({confident_pct:.1f}%)",
                    "📊",
                )
