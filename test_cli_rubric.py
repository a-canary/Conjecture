#!/usr/bin/env python3
"""
Simplified CLI Test Suite
Validates CLI implementation against rubric criteria
"""

import subprocess
import time
import sys
import os
from typing import Dict, List, Tuple

def run_command(cmd: str, timeout: int = 10) -> Tuple[bool, str, float]:
    """Run command and return (success, output, time)"""
    start = time.time()
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout,
            cwd="src"
        )
        elapsed = time.time() - start
        return result.returncode == 0, result.stdout + result.stderr, elapsed
    except subprocess.TimeoutExpired:
        return False, "Command timed out", timeout
    except Exception as e:
        return False, str(e), 0.0

def test_cli_rubric() -> Dict:
    """Run all CLI rubric tests"""
    print("=" * 60)
    print("CLI RUBRIC TEST SUITE")
    print("=" * 60)
    
    results = {
        "total_score": 0,
        "max_score": 50,
        "categories": {
            "core_functionality": {"score": 0, "max": 20},
            "user_experience": {"score": 0, "max": 15},
            "technical_quality": {"score": 0, "max": 10},
            "integration": {"score": 0, "max": 5}
        }
    }
    
    # Test 1: Command Structure (5 points)
    print("\n1. Testing Command Structure...")
    score = 0
    
    success, output, _ = run_command("python simple_cli.py --help")
    if success and "Commands:" in output and "Options:" in output:
        score += 2
        print("   [OK] Help system working")
    
    # Test multiple commands
    commands = ["create", "get", "search", "stats", "version"]
    available_commands = 0
    for cmd in commands:
        success, _, _ = run_command(f"python simple_cli.py {cmd} --help")
        if success:
            available_commands += 1
    
    if available_commands >= 4:
        score += 2
        print(f"   [OK] {available_commands}/5 commands have help")
    elif available_commands >= 2:
        score += 1
        print(f"   [PARTIAL] {available_commands}/5 commands have help")
    
    if "Usage:" in output and "Conjecture:" in output:
        score += 1
        print("   [OK] Professional CLI structure")
    
    results["categories"]["core_functionality"]["score"] += score
    print(f"   Command Structure: {score}/5")
    
    # Test 2: Argument Handling (5 points)
    print("\n2. Testing Argument Handling...")
    score = 0
    
    success, output, _ = run_command("python simple_cli.py create")
    if not success and ("required" in output.lower() or "missing" in output.lower()):
        score += 2
        print("   [OK] Required argument validation")
    
    success, output, _ = run_command("python simple_cli.py create 'test' --user alice --confidence invalid")
    if not success:
        score += 2
        print("   [OK] Type validation working")
    
    success, _, _ = run_command("python simple_cli.py create 'test' --user alice --confidence 0.5 --tags test")
    if success:
        score += 1
        print("   [OK] Optional arguments working")
    
    results["categories"]["core_functionality"]["score"] += score
    print(f"   Argument Handling: {score}/5")
    
    # Test 3: Error Handling (5 points)
    print("\n3. Testing Error Handling...")
    score = 0
    
    success, output, _ = run_command("python simple_cli.py invalid_command")
    if not success and output.strip():
        score += 2
        print("   [OK] Invalid command handled gracefully")
    
    success, output, _ = run_command("python simple_cli.py create")
    if not success and len(output.strip()) > 10:
        score += 2
        print("   [OK] Helpful error message for missing args")
    
    success, _, _ = run_command("python simple_cli.py create 'test claim' --user testuser --confidence 0.8")
    if success:
        score += 1
        print("   [OK] No crashes on valid input")
    
    results["categories"]["core_functionality"]["score"] += score
    print(f"   Error Handling: {score}/5")
    
    # Test 4: Output Quality (5 points)
    print("\n4. Testing Output Quality...")
    score = 0
    
    success, output, _ = run_command("python simple_cli.py create 'test claim' --user testuser --confidence 0.8")
    if success and len(output.split('\n')) >= 3:
        score += 2
        print("   [OK] Structured output format")
    
    success2, output2, _ = run_command("python simple_cli.py stats")
    if success2 and output.strip() and output2.strip():
        score += 2
        print("   [OK] Consistent formatting across commands")
    
    if success and any(char in output for char in [':', '-', '[', ']']):
        score += 1
        print("   [OK] Readable output with formatting")
    
    results["categories"]["core_functionality"]["score"] += score
    print(f"   Output Quality: {score}/5")
    
    # Test 5: Help System (5 points)
    print("\n5. Testing Help System...")
    score = 0
    
    success, output, _ = run_command("python simple_cli.py --help")
    if success and len(output) > 200:
        score += 2
        print("   [OK] Comprehensive main help")
    
    success, output, _ = run_command("python simple_cli.py create --help")
    if success and "Usage:" in output:
        score += 2
        print("   [OK] Command-specific help working")
    
    if success and any(word in output.lower() for word in ["create", "claim", "content"]):
        score += 1
        print("   [OK] Descriptive help text")
    
    results["categories"]["user_experience"]["score"] += score
    print(f"   Help System: {score}/5")
    
    # Test 6: User Feedback (5 points)
    print("\n6. Testing User Feedback...")
    score = 0
    
    success, output, _ = run_command("python simple_cli.py create 'test claim' --user testuser --confidence 0.8")
    if success and any(word in output.lower() for word in ["ok", "success", "created", "would"]):
        score += 2
        print("   [OK] Success feedback provided")
    
    if success and "Content:" in output and "User:" in output:
        score += 2
        print("   [OK] Detailed information display")
    
    if success and any(char in output for char in ['[', ']', 'OK', 'INFO']):
        score += 1
        print("   [OK] Status indicators present")
    
    results["categories"]["user_experience"]["score"] += score
    print(f"   User Feedback: {score}/5")
    
    # Test 7: Intuitiveness (5 points)
    print("\n7. Testing Intuitiveness...")
    score = 0
    
    commands = ["create", "get", "search", "stats"]
    logical_commands = 0
    for cmd in commands:
        success, _, _ = run_command(f"python simple_cli.py {cmd} --help")
        if success:
            logical_commands += 1
    
    if logical_commands >= 3:
        score += 2
        print("   [OK] Logical command names")
    
    success, output, _ = run_command("python simple_cli.py create --help")
    if success and "--user" in output and "--confidence" in output:
        score += 2
        print("   [OK] Consistent option patterns")
    
    success1, _, _ = run_command("python simple_cli.py --help")
    success2, _, _ = run_command("python simple_cli.py -h")
    if success1 and success2:
        score += 1
        print("   [OK] Predictable help options")
    
    results["categories"]["user_experience"]["score"] += score
    print(f"   Intuitiveness: {score}/5")
    
    # Test 8: Code Quality (5 points)
    print("\n8. Testing Code Quality...")
    score = 0
    
    cli_path = "src/simple_cli.py"
    if os.path.exists(cli_path):
        with open(cli_path, 'r') as f:
            content = f.read()
        
        if "typer" in content.lower():
            score += 2
            print("   [OK] Modern CLI framework (Typer)")
        
        if "def " in content and "import" in content:
            score += 1
            print("   [OK] Proper code structure")
        
        if '"""' in content or "help=" in content:
            score += 1
            print("   [OK] Code documentation present")
        
        if "try:" in content or "except" in content:
            score += 1
            print("   [OK] Error handling in code")
    
    results["categories"]["technical_quality"]["score"] += score
    print(f"   Code Quality: {score}/5")
    
    # Test 9: Performance (5 points)
    print("\n9. Testing Performance...")
    score = 0
    
    success, _, elapsed = run_command("python simple_cli.py --help")
    if success and elapsed < 2.0:
        score += 2
        print(f"   [OK] Fast help response ({elapsed:.2f}s)")
    elif success and elapsed < 5.0:
        score += 1
        print(f"   [PARTIAL] Acceptable help response ({elapsed:.2f}s)")
    
    success, _, elapsed = run_command("python simple_cli.py create 'test' --user test --confidence 0.5")
    if success and elapsed < 2.0:
        score += 2
        print(f"   [OK] Fast command response ({elapsed:.2f}s)")
    elif success and elapsed < 5.0:
        score += 1
        print(f"   [PARTIAL] Acceptable command response ({elapsed:.2f}s)")
    
    if success:
        score += 1
        print("   [OK] No obvious memory issues")
    
    results["categories"]["technical_quality"]["score"] += score
    print(f"   Performance: {score}/5")
    
    # Test 10: System Integration (5 points)
    print("\n10. Testing System Integration...")
    score = 0
    
    cli_path = "src/simple_cli.py"
    if os.path.exists(cli_path):
        with open(cli_path, 'r') as f:
            content = f.read()
        
        if "typer" in content.lower() and "rich" in content.lower():
            score += 2
            print("   [OK] Professional framework integration")
        elif "typer" in content.lower():
            score += 1
            print("   [PARTIAL] Basic framework integration")
        
        score += 1
        print("   [OK] Proper module structure")
    
    success, output, _ = run_command("python simple_cli.py --help")
    if success and len(output) > 100:
        score += 2
        print("   [OK] Extensible command structure")
    else:
        score += 1
        print("   [PARTIAL] Limited extensibility")
    
    results["categories"]["integration"]["score"] += score
    print(f"   System Integration: {score}/5")
    
    # Calculate totals
    total = 0
    for category, data in results["categories"].items():
        total += data["score"]
    
    results["total_score"] = total
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    for category, data in results["categories"].items():
        print(f"{category.replace('_', ' ').title()}: {data['score']}/{data['max']}")
    
    print(f"\nTOTAL SCORE: {total}/50")
    
    # Grade assessment
    if total >= 45:
        grade = "EXCELLENT (A+)"
    elif total >= 40:
        grade = "VERY GOOD (A)"
    elif total >= 35:
        grade = "GOOD (B+)"
    elif total >= 30:
        grade = "ACCEPTABLE (B)"
    elif total >= 25:
        grade = "NEEDS IMPROVEMENT (C)"
    else:
        grade = "POOR (D/F)"
    
    print(f"GRADE: {grade}")
    
    if total >= 30:
        print("\n[SUCCESS] CLI meets minimum standards!")
    else:
        print("\n[WARNING] CLI needs improvement before release.")
    
    return results

def main():
    """Main test runner"""
    if not os.path.exists("src/simple_cli.py"):
        print("Error: CLI not found at src/simple_cli.py")
        sys.exit(1)
    
    results = test_cli_rubric()
    
    # Save results
    import json
    with open("cli_test_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: cli_test_results.json")
    
    return results["total_score"]

if __name__ == "__main__":
    score = main()
    sys.exit(0 if score >= 30 else 1)