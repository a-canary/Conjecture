"""
Vector Database Evaluation Runner
Tests all available vector database implementations and generates comparison report
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.db_evaluation_framework import DatabaseEvaluator, compare_evaluations
from src.data.mock_chroma import MockChromaDB
from src.data.vectors.chromadb_integration import (
    ChromaDBIntegration,
    create_chromadb_config,
)
from src.data.vectors.faiss_integration import FaissIntegration, create_faiss_config


class MockChromaDBWrapper:
    """Wrapper to make MockChromaDB compatible with evaluation framework"""

    def __init__(self, config):
        self.db = MockChromaDB(config.get("path", "./data/mock_eval.json"))

    def add_claim(self, claim):
        return self.db.add_claim(claim)

    def get_claim(self, claim_id):
        return self.db.get_claim(claim_id)

    def search_similar(self, query, limit=5):
        # MockChromaDB doesn't return similarity scores, so we fake them
        results = self.db.search_by_content(query, limit)
        return [(claim, 0.8) for claim in results]  # Fake similarity score

    def filter_claims(self, **filters):
        return self.db.filter_claims(**filters)

    def batch_add_claims(self, claims):
        success = True
        for claim in claims:
            if not self.db.add_claim(claim):
                success = False
        return success

    def get_stats(self):
        return {
            "total_claims": self.db.get_claim_count(),
            "type": "MockChromaDB",
        }

    def clear_all(self):
        return self.db.clear_all()

    def close(self):
        pass


def run_evaluation():
    """Run complete evaluation of all available databases"""
    print("=" * 80)
    print("Conjecture VECTOR DATABASE EVALUATION")
    print("=" * 80)
    print("Success Criteria: ≥35/40 points, <100ms queries with 10k claims")
    print("=" * 80)

    evaluator = DatabaseEvaluator()
    evaluations = []

    # Test 1: MockChromaDB (baseline)
    print("\n1. Testing MockChromaDB (baseline)...")
    try:
        mock_config = {"path": "./data/eval_mock.json"}
        mock_db = MockChromaDBWrapper(mock_config)
        mock_eval = evaluator.evaluate_database(mock_db, "MockChromaDB (Baseline)")
        evaluations.append(mock_eval)
        evaluator.print_evaluation_report(mock_eval)
    except Exception as e:
        print(f"MockChromaDB evaluation failed: {e}")

    # Test 2: ChromaDB
    print("\n2. Testing ChromaDB...")
    try:
        chroma_config = create_chromadb_config(
            persist_directory="./data/eval_chroma", collection_name="Conjecture_eval"
        )
        chroma_db = ChromaDBIntegration(chroma_config)
        chroma_eval = evaluator.evaluate_database(chroma_db, "ChromaDB")
        evaluations.append(chroma_eval)
        evaluator.print_evaluation_report(chroma_eval)
        chroma_db.close()
    except Exception as e:
        print(f"ChromaDB evaluation failed: {e}")
        print("Try installing with: pip install chromadb")

    # Test 3: Faiss
    print("\n3. Testing Faiss...")
    try:
        faiss_config = create_faiss_config(
            index_path="./data/eval_faiss.index",
            index_type="flat",  # Start with flat for exact search
            embedding_dim=384,
        )
        faiss_db = FaissIntegration(faiss_config)
        faiss_eval = evaluator.evaluate_database(faiss_db, "Faiss")
        evaluations.append(faiss_eval)
        evaluator.print_evaluation_report(faiss_eval)
        faiss_db.close()
    except Exception as e:
        print(f"Faiss evaluation failed: {e}")
        print("Try installing with: pip install faiss-cpu sentence-transformers")

    # Generate comparison report
    if evaluations:
        print("\n" + "=" * 80)
        print("FINAL COMPARISON AND RECOMMENDATIONS")
        print("=" * 80)

        rankings = compare_evaluations(evaluations)

        # Find best performer
        best_db = max(evaluations, key=lambda x: x.percentage)

        print(f"\n🏆 BEST PERFORMER: {best_db.database_name}")
        print(
            f"   Score: {best_db.percentage:.1f}% ({best_db.total_score}/{best_db.max_score})"
        )
        print(f"   Recommendation: {best_db.recommendation}")

        # Performance comparison
        print(f"\n📊 PERFORMANCE COMPARISON:")
        print("-" * 80)
        print(
            f"{'Database':<20} {'Score':<8} {'Status':<15} {'Query Speed':<12} {'Memory':<10}"
        )
        print("-" * 80)

        for eval in evaluations:
            # Get performance metrics
            query_time = eval.performance_summary.get("query_time_10_results", 0)
            query_speed = f"{query_time * 1000:.1f}ms" if query_time > 0 else "N/A"

            # Determine status
            if eval.percentage >= 35:
                status = "✅ MEETS CRITERIA"
            elif eval.percentage >= 30:
                status = "⚠️  CLOSE"
            else:
                status = "❌ BELOW CRITERIA"

            print(
                f"{eval.database_name:<20} {eval.percentage:>5.1f}%  {status:<15} {query_speed:<12} {'N/A':<10}"
            )

        # Recommendations based on results
        print(f"\n🎯 SPECIFIC RECOMMENDATIONS:")

        meets_criteria = [e for e in evaluations if e.percentage >= 35]

        if meets_criteria:
            print("✅ Databases meeting success criteria:")
            for eval in meets_criteria:
                print(f"   - {eval.database_name}: {eval.recommendation}")

            best_meeting = max(meets_criteria, key=lambda x: x.percentage)
            print(f"\n🎉 RECOMMENDED CHOICE: {best_meeting.database_name}")
            print(
                f"   Reason: Highest score ({best_meeting.percentage:.1f}%) among databases meeting criteria"
            )
        else:
            print("❌ No databases meet the full success criteria")
            closest = max(evaluations, key=lambda x: x.percentage)
            print(f"\n🔧 CLOSEST OPTION: {closest.database_name}")
            print(
                f"   Score: {closest.percentage:.1f}% ({35 - closest.percentage:.1f}% below criteria)"
            )

            # Identify what needs improvement
            low_scores = [r for r in closest.results if not r.passed_threshold]
            if low_scores:
                print("   Areas needing improvement:")
                for result in low_scores:
                    print(f"     - {result.metric_name}: {result.details}")

        # Installation recommendations
        print(f"\n📦 INSTALLATION RECOMMENDATIONS:")

        chroma_evals = [e for e in evaluations if "ChromaDB" in e.database_name]
        faiss_evals = [e for e in evaluations if "Faiss" in e.database_name]

        if chroma_evals:
            print("   ChromaDB: pip install chromadb")
        if faiss_evals:
            print("   Faiss: pip install faiss-cpu sentence-transformers")

        # Next steps
        print(f"\n🚀 NEXT STEPS:")
        if meets_criteria:
            print("   1. Install recommended database")
            print("   2. Update configuration to use production database")
            print("   3. Run integration tests with real data")
            print("   4. Move to Phase 2: LLM API Integration")
        else:
            print("   1. Install missing dependencies")
            print("   2. Re-run evaluation")
            print("   3. Consider custom implementation if needed")
            print("   4. Continue with Mock implementation for development")

    else:
        print("❌ No evaluations completed successfully")
        print("\nTroubleshooting:")
        print("1. Ensure Python dependencies are installed:")
        print("   pip install chromadb faiss-cpu sentence-transformers numpy")
        print("2. Check that data directory is writable")
        print("3. Ensure enough disk space for test data")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_evaluation()
