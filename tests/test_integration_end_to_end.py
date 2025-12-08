"""
End-to-End Integration Test for Conjecture System
Tests complete workflow from exploration to evaluation
"""

import asyncio
import sys
import os
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.core.models import Claim, ClaimType, ClaimState
from src.config.unified_config import UnifiedConfig as Config
# from enhanced_conjecture import EnhancedConjecture, ExplorationResult  # Commented out as file doesn't exist


class EndToEndIntegrationTest:
    """Complete end-to-end integration test for Conjecture"""

    def __init__(self):
        self.test_results = []
        self.temp_dir = None

    async def setup(self):
        """Setup test environment"""
        print("Setting up integration test environment...")

        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp(prefix="conjecture_test_")
        print(f"Created temporary directory: {self.temp_dir}")

        # Setup test configuration
        self.config = Config()
        self.config.database_type = "mock"  # Use mock for testing
        self.config.llm_provider = "mock"  # Use mock for testing

        print("Setup complete")

    async def teardown(self):
        """Clean up test environment"""
        print("Cleaning up test environment...")

        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"Removed temporary directory: {self.temp_dir}")

        print("Cleanup complete")

    async def test_01_basic_initialization(self):
        """Test basic system initialization"""
        print("\n=== Test 1: Basic Initialization ===")

        try:
            # Test configuration creation
            config = Config()
            assert config is not None
            assert hasattr(config, "database_type")
            assert hasattr(config, "llm_provider")
            print("✅ Configuration creation successful")

            # Test claim model creation
            claim = Claim(
                id="test_001",
                content="Integration test claim with sufficient length for validation",
                confidence=0.85,
                type=[ClaimType.CONCEPT],
                tags=["test", "integration"],
            )
            assert claim.id == "test_001"
            assert claim.confidence == 0.85
            assert ClaimType.CONCEPT in claim.type
            print("✅ Claim model creation successful")

            # Test claim relationships
            claim2 = Claim(
                id="test_002",
                content="Second integration test claim for relationship testing",
                confidence=0.75,
                type=[ClaimType.EXAMPLE],
            )

            claim.add_supports(claim2.id)
            claim2.add_support(claim.id)
            assert claim2.id in claim.supports
            assert claim.id in claim2.supported_by
            print("✅ Claim relationship management successful")

            self.test_results.append(
                ("basic_initialization", True, "All basic components working")
            )
            return True

        except Exception as e:
            error_msg = f"Basic initialization failed: {e}"
            print(f"❌ {error_msg}")
            self.test_results.append(("basic_initialization", False, error_msg))
            return False

    async def test_02_enhanced_conjecture_creation(self):
        """Test Enhanced Conjecture creation and basic functionality"""
        print("\n=== Test 2: Enhanced Conjecture Creation ===")

        try:
            # Test with mock configuration
            config = Config()
            config.database_type = "mock"
            config.llm_provider = "mock"

            # Create Enhanced Conjecture instance
            cf = EnhancedConjecture(config=config)
            assert cf is not None
            assert cf.config == config
            print("✅ Enhanced Conjecture creation successful")

            # Test statistics access
            stats = cf.get_statistics()
            assert "config" in stats
            assert "services_running" in stats
            print("✅ Statistics access successful")

            self.test_results.append(
                ("enhanced_conjecture_creation", True, "Enhanced Conjecture working")
            )
            return True

        except Exception as e:
            error_msg = f"Enhanced Conjecture creation failed: {e}"
            print(f"❌ {error_msg}")
            self.test_results.append(("enhanced_conjecture_creation", False, error_msg))
            return False

    async def test_03_claim_creation_and_management(self):
        """Test claim creation and management functionality"""
        print("\n=== Test 3: Claim Creation and Management ===")

        try:
            config = Config()
            config.database_type = "mock"
            config.llm_provider = "mock"

            async with EnhancedConjecture(config=config) as cf:
                # Test basic claim creation
                claim = await cf.add_claim(
                    content="Test claim for integration testing",
                    confidence=0.8,
                    claim_type="concept",
                    tags=["integration", "test"],
                    auto_evaluate=False,  # Disable auto-evaluation for simpler testing
                )

                assert claim is not None
                assert claim.content == "Test claim for integration testing"
                assert claim.confidence == 0.8
                assert ClaimType.CONCEPT in claim.type
                print("✅ Basic claim creation successful")

                # Test claim relationships
                claim2 = await cf.add_claim(
                    content="Supporting claim for integration testing",
                    confidence=0.75,
                    claim_type="example",
                    auto_evaluate=False,
                )

                claim.add_supports(claim2.id)
                claim2.add_support(claim.id)
                assert claim2.id in claim.supports
                assert claim.id in claim2.supported_by
                print("✅ Claim relationship management successful")

                # Test confidence updates
                original_confidence = claim.confidence
                claim.update_confidence(0.9)
                assert claim.confidence == 0.9
                assert claim.confidence != original_confidence
                print("✅ Confidence update successful")

                self.test_results.append(
                    ("claim_management", True, "Claim creation and management working")
                )
                return True

        except Exception as e:
            error_msg = f"Claim management failed: {e}"
            print(f"❌ {error_msg}")
            self.test_results.append(("claim_management", False, error_msg))
            return False

    async def test_04_context_formatting(self):
        """Test context formatting and claim representation"""
        print("\n=== Test 4: Context Formatting ===")

        try:
            # Create test claims
            claims = [
                Claim(
                    id="ctx_test_001",
                    content="Context formatting test claim one",
                    confidence=0.85,
                    type=[ClaimType.CONCEPT],
                    tags=["context", "formatting"],
                ),
                Claim(
                    id="ctx_test_002",
                    content="Context formatting test claim two",
                    confidence=0.75,
                    type=[ClaimType.EXAMPLE],
                    tags=["context", "formatting", "example"],
                ),
            ]

            # Test context formatting
            formatted_contexts = []
            for claim in claims:
                formatted = claim.format_for_context()
                formatted_contexts.append(formatted)
                assert isinstance(formatted, str)
                assert claim.id in formatted
                assert str(claim.confidence) in formatted
                print(f"✅ Formatted context: {formatted[:50]}...")

            # Test Chroma metadata conversion
            metadata = claims[0].to_chroma_metadata()
            assert "confidence" in metadata
            assert "type" in metadata
            assert "tags" in metadata
            print("✅ Chroma metadata conversion successful")

            self.test_results.append(
                ("context_formatting", True, "Context formatting working")
            )
            return True

        except Exception as e:
            error_msg = f"Context formatting failed: {e}"
            print(f"❌ {error_msg}")
            self.test_results.append(("context_formatting", False, error_msg))
            return False

    async def test_05_tool_validation(self):
        """Test tool validation functionality"""
        print("\n=== Test 5: Tool Validation ===")

        try:
            from tests.test_tool_validator_simple import ToolValidator

            validator = ToolValidator()

            # Test safe code validation
            safe_code = '''
def execute(param: str) -> dict:
    """Execute a safe operation"""
    return {"success": True, "result": param}
'''

            is_valid, issues = validator.validate_tool_code(safe_code)
            assert is_valid, f"Safe code should be valid: {issues}"
            print("✅ Safe code validation successful")

            # Test unsafe code detection
            unsafe_code = '''
import os
def execute(param: str) -> dict:
    """Execute with dangerous import"""
    return {"success": True, "result": param}
'''

            is_valid, issues = validator.validate_tool_code(unsafe_code)
            assert not is_valid, "Unsafe code should not be valid"
            assert len(issues) > 0
            print("✅ Unsafe code detection successful")

            self.test_results.append(
                ("tool_validation", True, "Tool validation working")
            )
            return True

        except Exception as e:
            error_msg = f"Tool validation failed: {e}"
            print(f"❌ {error_msg}")
            self.test_results.append(("tool_validation", False, error_msg))
            return False

    async def test_06_performance_benchmarks(self):
        """Test performance benchmark components"""
        print("\n=== Test 6: Performance Benchmarks ===")

        try:
            from tests.performance_benchmarks_final import PerformanceBenchmark

            benchmark = PerformanceBenchmark()

            # Test claim creation benchmark
            result = benchmark.benchmark_claim_creation(100)
            assert "execution_time" in result
            assert "claims_per_second" in result
            assert result["num_claims"] == 100
            print(
                f"✅ Performance benchmark successful: {result['claims_per_second']:.0f} claims/sec"
            )

            self.test_results.append(
                ("performance_benchmarks", True, "Performance benchmarks working")
            )
            return True

        except Exception as e:
            error_msg = f"Performance benchmarks failed: {e}"
            print(f"❌ {error_msg}")
            self.test_results.append(("performance_benchmarks", False, error_msg))
            return False

    async def test_07_exploration_result_handling(self):
        """Test exploration result creation and handling"""
        print("\n=== Test 7: Exploration Result Handling ===")

        try:
            # Create test claims
            test_claims = [
                Claim(
                    id="exp_001",
                    content="Exploration test claim one",
                    confidence=0.8,
                    type=[ClaimType.CONCEPT],
                ),
                Claim(
                    id="exp_002",
                    content="Exploration test claim two",
                    confidence=0.75,
                    type=[ClaimType.EXAMPLE],
                ),
            ]

            # Create exploration result
            result = ExplorationResult(
                query="test exploration",
                claims=test_claims,
                total_found=len(test_claims),
                search_time=0.1,
                confidence_threshold=0.5,
                max_claims=10,
            )

            assert result.query == "test exploration"
            assert len(result.claims) == 2
            assert result.total_found == 2
            print("✅ Exploration result creation successful")

            # Test summary generation
            summary = result.summary()
            assert "test exploration" in summary
            assert "2 claims" in summary
            print("✅ Summary generation successful")

            self.test_results.append(
                ("exploration_result", True, "Exploration result handling working")
            )
            return True

        except Exception as e:
            error_msg = f"Exploration result handling failed: {e}"
            print(f"❌ {error_msg}")
            self.test_results.append(("exploration_result", False, error_msg))
            return False

    async def run_all_tests(self):
        """Run all integration tests"""
        print("Starting Conjecture End-to-End Integration Tests")
        print("=" * 60)

        await self.setup()

        try:
            # Run tests in order
            tests = [
                self.test_01_basic_initialization,
                self.test_02_enhanced_conjecture_creation,
                self.test_03_claim_creation_and_management,
                self.test_04_context_formatting,
                self.test_05_tool_validation,
                self.test_06_performance_benchmarks,
                self.test_07_exploration_result_handling,
            ]

            for test in tests:
                try:
                    await test()
                except Exception as e:
                    print(f"⚠️  Test {test.__name__} failed with exception: {e}")
                    continue

            await self.teardown()

            # Generate final report
            await self.generate_report()

            return self.get_overall_result()

        except Exception as e:
            print(f"❌ Critical error in test suite: {e}")
            await self.teardown()
            return False

    async def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print("CONJECTURE END-TO-END INTEGRATION TEST REPORT")
        print("=" * 60)

        passed = 0
        failed = 0

        for test_name, success, message in self.test_results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} {test_name}: {message}")

            if success:
                passed += 1
            else:
                failed += 1

        print("-" * 60)
        print(f"Total Tests: {len(self.test_results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(
            f"Success Rate: {passed / len(self.test_results) * 100:.1f}%"
            if self.test_results
            else "0%"
        )

        # System information
        import platform

        print(f"\nSystem: {platform.system()} {platform.release()}")
        print(f"Python: {platform.python_version()}")

        print("=" * 60)

    def get_overall_result(self):
        """Get overall test result"""
        if not self.test_results:
            return False

        passed = sum(1 for _, success, _ in self.test_results if success)
        return passed == len(self.test_results)


async def main():
    """Main test runner"""
    tester = EndToEndIntegrationTest()
    success = await tester.run_all_tests()

    if success:
        print("\n🎉 All integration tests passed! Conjecture is working correctly.")
        return 0
    else:
        print("\n💥 Some integration tests failed. Please review the report above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
