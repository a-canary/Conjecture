#!/usr/bin/env python3
"""
Real-time Statistics Generator for Conjecture Project

Generates comprehensive statistics including:
- Project metrics from ANALYSIS.md
- Current benchmark scores from all test models
- System configuration status
- Test coverage and performance metrics
- Real implementation state (no simulated results)

Output: ./STATS.yaml
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import specialized modules
try:
    from .benchmark_collector import BenchmarkCollector
    from .test_analyzer import TestAnalyzer
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append(str(Path(__file__).parent))
    from benchmark_collector import BenchmarkCollector
    from test_analyzer import TestAnalyzer

class ProjectStatsGenerator:
    """Generates real-time project statistics"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.stats = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "generator": "src/stats/main.py",
                "project_root": str(self.project_root)
            }
        }

        # Initialize specialized analyzers
        self.benchmark_collector = BenchmarkCollector(self.project_root)
        self.test_analyzer = TestAnalyzer(self.project_root)

    def generate_all_stats(self) -> Dict[str, Any]:
        """Generate all statistics"""
        print("Generating real-time project statistics...")

        # Core statistics using specialized analyzers
        self.stats.update({
            "project_metrics": self.get_project_metrics(),
            "test_results": self.test_analyzer.analyze_tests(),
            "benchmark_scores": self.benchmark_collector.collect_all_benchmarks(),
            "benchmark_summary": self.benchmark_collector.get_benchmark_summary(),
            "test_health": self.test_analyzer.get_test_health_score(),
            "system_configuration": self.get_system_configuration(),
            "infrastructure_status": self.get_infrastructure_status(),
            "implementation_state": self.get_implementation_state()
        })

        return self.stats

    def get_project_metrics(self) -> Dict[str, Any]:
        """Get project metrics from ANALYSIS.md and file system"""
        print("  Analyzing project metrics...")

        metrics = {}

        # Try to read ANALYSIS.md for baseline metrics
        analysis_file = self.project_root / "ANALYSIS.md"
        if analysis_file.exists():
            try:
                with open(analysis_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Extract key metrics with improved parsing
                for line in content.split('\n'):
                    if ':' in line and not line.strip().startswith('#'):
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()

                        # Helper function to parse different number formats
                        def parse_numeric(value_str):
                            """Parse various numeric formats including '9.8/10', percentages, etc."""
                            value_str = value_str.strip()

                            # Handle fraction format like '9.8/10'
                            if '/' in value_str and not value_str.startswith('/'):
                                try:
                                    numerator, denominator = value_str.split('/', 1)
                                    num = float(numerator.strip())
                                    den = float(denominator.strip())
                                    if den != 0:
                                        return round(num / den * 100, 1)  # Convert to percentage
                                except:
                                    pass

                            # Handle percentage format
                            if '%' in value_str:
                                try:
                                    return float(value_str.replace('%', '').strip())
                                except:
                                    pass

                            # Handle range like "50 / 51 (98% core tests passing)"
                            if ' / ' in value_str and '(' in value_str:
                                try:
                                    parts = value_str.split()
                                    for i, part in enumerate(parts):
                                        if part == '/' and i > 0 and i < len(parts) - 1:
                                            numerator = float(parts[i-1])
                                            denominator = float(parts[i+1].strip('()'))
                                            if denominator != 0:
                                                return round(numerator / denominator * 100, 1)
                                except:
                                    pass

                            # Handle simple numbers
                            try:
                                # Extract first number from string
                                import re
                                match = re.search(r'-?\d+\.?\d*', value_str)
                                if match:
                                    return float(match.group())
                            except:
                                pass

                            return None

                        # Clean up metrics with improved parsing
                        if 'code_files' in key:
                            parsed = parse_numeric(value)
                            if parsed is not None:
                                metrics['code_files'] = int(parsed)
                        elif 'docs_files' in key:
                            parsed = parse_numeric(value)
                            if parsed is not None:
                                metrics['docs_files'] = int(parsed)
                        elif 'test_coverage' in key:
                            parsed = parse_numeric(value)
                            if parsed is not None:
                                metrics['test_coverage'] = parsed
                        elif 'test_pass' in key and 'test_pass_rate' not in key:
                            parsed = parse_numeric(value)
                            if parsed is not None:
                                metrics['test_pass'] = int(parsed)
                        elif 'code_quality_score' in key:
                            parsed = parse_numeric(value)
                            if parsed is not None:
                                metrics['code_quality_score'] = parsed
                        elif 'security_score' in key:
                            parsed = parse_numeric(value)
                            if parsed is not None:
                                metrics['security_score'] = parsed
                        elif 'time_required' in key:
                            # Extract time in seconds
                            try:
                                import re
                                match = re.search(r'(\d+\.?\d*)\s*sec', value.lower())
                                if match:
                                    metrics['time_required'] = float(match.group(1))
                                else:
                                    parsed = parse_numeric(value)
                                    if parsed is not None:
                                        metrics['time_required'] = parsed
                            except:
                                pass
                        elif 'uptime' in key:
                            parsed = parse_numeric(value)
                            if parsed is not None:
                                metrics['uptime'] = parsed
                        elif 'error_rate' in key:
                            parsed = parse_numeric(value)
                            if parsed is not None:
                                metrics['error_rate'] = parsed
                        elif 'ci_cd_readiness' in key:
                            parsed = parse_numeric(value)
                            if parsed is not None:
                                metrics['ci_cd_readiness'] = parsed
                        elif 'test_collection_success' in key:
                            parsed = parse_numeric(value)
                            if parsed is not None:
                                metrics['test_collection_success'] = parsed
                        elif 'test_pass_rate' in key:
                            parsed = parse_numeric(value)
                            if parsed is not None:
                                metrics['test_pass_rate'] = parsed
                        elif 'linting_errors' in key:
                            parsed = parse_numeric(value)
                            if parsed is not None:
                                metrics['linting_errors'] = int(parsed)
                        elif 'orphaned_files' in key:
                            parsed = parse_numeric(value)
                            if parsed is not None:
                                metrics['orphaned_files'] = int(parsed)
                        elif 'reachable_files' in key:
                            parsed = parse_numeric(value)
                            if parsed is not None:
                                metrics['reachable_files'] = int(parsed)
                        elif 'dead_code_percentage' in key:
                            parsed = parse_numeric(value)
                            if parsed is not None:
                                metrics['dead_code_percentage'] = parsed
                        elif 'static_analysis_integration' in key:
                            parsed = parse_numeric(value)
                            if parsed is not None:
                                metrics['static_analysis_integration'] = parsed
                        elif 'pytest_configuration' in key:
                            parsed = parse_numeric(value)
                            if parsed is not None:
                                metrics['pytest_configuration'] = parsed
                        elif 'pytest_runtime' in key:
                            # Extract runtime in seconds
                            try:
                                import re
                                match = re.search(r'(\d+\.?\d*)s?', value.lower())
                                if match:
                                    metrics['pytest_runtime'] = float(match.group(1))
                            except:
                                pass
                        elif 'e2e_test_failures' in key:
                            parsed = parse_numeric(value)
                            if parsed is not None:
                                metrics['e2e_test_failures'] = int(parsed)

            except Exception as e:
                print(f"    Warning: Could not parse ANALYSIS.md: {e}")
                import traceback
                print(f"    Detailed error: {traceback.format_exc()}")

        # File system metrics (real-time)
        try:
            src_dir = self.project_root / "src"
            if src_dir.exists():
                python_files = list(src_dir.rglob("*.py"))
                metrics['actual_code_files'] = len(python_files)

                total_lines = 0
                for py_file in python_files:
                    try:
                        with open(py_file, 'r', encoding='utf-8') as f:
                            total_lines += len(f.readlines())
                    except:
                        pass
                metrics['total_lines_of_code'] = total_lines
        except Exception as e:
            print(f"    Warning: Could not analyze file system: {e}")

        return metrics

    def get_test_results(self) -> Dict[str, Any]:
        """Get current test results"""
        print("  Running test analysis...")

        test_results = {}

        try:
            # Run pytest to get real test results
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                "--tb=no",
                "-q",
                "--collect-only"
            ], capture_output=True, text=True, cwd=self.project_root, timeout=30)

            if result.returncode == 0:
                output = result.stdout
                if "test session starts" in output:
                    # Extract test count
                    for line in output.split('\n'):
                        if "collected" in line.lower() and "items" in line.lower():
                            try:
                                count = int(line.split()[1])
                                test_results['tests_collected'] = count
                                break
                            except:
                                pass
            else:
                test_results['collection_error'] = result.stderr[:200]

        except Exception as e:
            test_results['collection_error'] = str(e)

        # Try to run a quick test subset for pass rate
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                "tests/test_claim_models.py",
                "tests/test_id_utilities.py",
                "-q",
                "--tb=no"
            ], capture_output=True, text=True, cwd=self.project_root, timeout=30)

            if result.returncode == 0:
                output = result.stdout
                # Parse passed/failed
                for line in output.split('\n'):
                    if 'passed' in line and 'failed' in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == 'passed' and i > 0:
                                test_results['core_tests_passed'] = int(parts[i-1])
                            elif part == 'failed' and i > 0:
                                test_results['core_tests_failed'] = int(parts[i-1])
                        break
        except Exception as e:
            test_results['core_test_error'] = str(e)

        return test_results

    def get_benchmark_scores(self) -> Dict[str, Any]:
        """Get current benchmark scores from real results"""
        print("  Analyzing benchmark results...")

        benchmark_scores = {}

        # Look for benchmark result files
        benchmark_dir = self.project_root / "src" / "benchmarking" / "cycle_results"
        if benchmark_dir.exists():
            try:
                result_files = list(benchmark_dir.glob("cycle_*_results.json"))
                latest_files = sorted(result_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]

                for result_file in latest_files:
                    try:
                        with open(result_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)

                        cycle_name = result_file.stem

                        if 'overall_score' in data:
                            benchmark_scores[f"{cycle_name}_overall"] = data['overall_score']

                        if 'scores' in data and isinstance(data['scores'], dict):
                            for benchmark, score_data in data['scores'].items():
                                if isinstance(score_data, dict) and 'overall_score' in score_data:
                                    benchmark_scores[f"{cycle_name}_{benchmark}"] = score_data['overall_score']
                                elif isinstance(score_data, (int, float)):
                                    benchmark_scores[f"{cycle_name}_{benchmark}"] = score_data

                    except Exception as e:
                        print(f"    Warning: Could not read {result_file}: {e}")

            except Exception as e:
                print(f"    Warning: Could not analyze benchmark directory: {e}")

        # Check for improved claim system results
        improved_results_file = self.project_root / "src" / "benchmarking" / "improved_claim_system_results.json"
        if improved_results_file.exists():
            try:
                with open(improved_results_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if 'performance_metrics' in data:
                    perf = data['performance_metrics']
                    benchmark_scores['improved_claim_direct_accuracy'] = perf.get('direct_accuracy', '?')
                    benchmark_scores['improved_claim_conjecture_accuracy'] = perf.get('conjecture_accuracy', '?')
                    benchmark_scores['improved_claim_improvement'] = perf.get('improvement', '?')

            except Exception as e:
                print(f"    Warning: Could not read improved claim results: {e}")

        return benchmark_scores

    def get_system_configuration(self) -> Dict[str, Any]:
        """Get current system configuration with comprehensive error handling"""
        print("  Analyzing system configuration...")

        config = {
            "config_exists": False,
            "configured_providers": 0,
            "local_providers": 0,
            "cloud_providers": 0,
            "primary_provider": "None",
            "primary_model": "None",
            "confidence_threshold": None,
            "config_errors": [],
            "config_warnings": []
        }

        # Check multiple possible config locations
        config_locations = [
            self.project_root / ".conjecture" / "config.json",
            self.project_root / "conjecture.json",
            self.project_root / "config.json"
        ]

        config_file = None
        for location in config_locations:
            if location.exists():
                config_file = location
                config["config_exists"] = True
                config["config_file_path"] = str(location)
                break

        if config_file is None:
            config["config_errors"].append("No configuration file found in expected locations")
            return config

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            # Extract key configuration details with validation
            if 'providers' in config_data and isinstance(config_data['providers'], list):
                providers = config_data['providers']
                config['configured_providers'] = len(providers)

                # Count local vs cloud providers
                local_count = 0
                cloud_count = 0

                for provider in providers:
                    if not isinstance(provider, dict):
                        config["config_warnings"].append(f"Invalid provider configuration: {provider}")
                        continue

                    # Check if provider is local (contains localhost, 127.0.0.1, or is_local flag)
                    is_local = provider.get('is_local', False)
                    if not is_local and 'url' in provider:
                        url = str(provider['url']).lower()
                        is_local = any(local in url for local in ['localhost', '127.0.0.1', '0.0.0.0'])

                    if is_local:
                        local_count += 1
                    else:
                        cloud_count += 1

                config['local_providers'] = local_count
                config['cloud_providers'] = cloud_count

                # Get primary provider (lowest priority)
                valid_providers = [p for p in providers if isinstance(p, dict) and 'priority' in p]
                if valid_providers:
                    try:
                        priorities = [(int(p.get('priority', 999)), p) for p in valid_providers]
                        priorities.sort()
                        primary = priorities[0][1]
                        config['primary_provider'] = primary.get('name', 'Unknown')
                        config['primary_model'] = primary.get('model', 'Unknown')
                    except Exception as e:
                        config["config_warnings"].append(f"Could not determine primary provider: {e}")
                else:
                    # Fallback: use first provider
                    if providers:
                        first_provider = providers[0]
                        if isinstance(first_provider, dict):
                            config['primary_provider'] = first_provider.get('name', 'Unknown')
                            config['primary_model'] = first_provider.get('model', 'Unknown')

            else:
                config["config_warnings"].append("No valid providers section found in configuration")

            # Extract confidence threshold with fallback
            threshold_sources = ['confidence_threshold', 'threshold', 'min_confidence']
            for source in threshold_sources:
                if source in config_data:
                    try:
                        threshold = float(config_data[source])
                        if 0 <= threshold <= 1:
                            config['confidence_threshold'] = threshold
                            break
                        else:
                            config["config_warnings"].append(f"Invalid confidence_threshold value: {threshold}")
                    except (ValueError, TypeError):
                        continue

            # Check for additional important settings
            important_settings = ['database_path', 'cache_dir', 'log_level', 'max_tokens', 'temperature']
            for setting in important_settings:
                if setting in config_data:
                    config[setting] = config_data[setting]

        except json.JSONDecodeError as e:
            config['config_errors'].append(f"Invalid JSON in configuration file: {e}")
        except UnicodeDecodeError as e:
            config['config_errors'].append(f"Configuration file encoding error: {e}")
        except Exception as e:
            config['config_errors'].append(f"Unexpected error reading configuration: {e}")

        return config

    def get_infrastructure_status(self) -> Dict[str, Any]:
        """Check infrastructure status with comprehensive analysis"""
        print("  Checking infrastructure status...")

        infrastructure = {
            "database_status": {
                "exists": False,
                "size_mb": 0,
                "writable": False,
                "tables": [],
                "errors": []
            },
            "directory_status": {},
            "python_environment": {
                "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "platform": sys.platform,
                "executable": sys.executable,
                "site_packages": []
            },
            "dependency_status": {
                "critical_packages": {},
                "missing_packages": [],
                "warnings": []
            }
        }

        # Check database infrastructure
        try:
            data_dir = self.project_root / "data"
            db_path = data_dir / "conjecture.db"

            infrastructure["database_status"]["exists"] = db_path.exists()

            if db_path.exists():
                try:
                    size_bytes = db_path.stat().st_size
                    infrastructure["database_status"]["size_mb"] = round(size_bytes / (1024 * 1024), 2)

                    # Test writability
                    test_file = data_dir / "test_write.tmp"
                    try:
                        test_file.write_text("test")
                        test_file.unlink()
                        infrastructure["database_status"]["writable"] = True
                    except Exception:
                        infrastructure["database_status"]["writable"] = False

                except Exception as e:
                    infrastructure["database_status"]["errors"].append(f"Database analysis error: {e}")
            else:
                infrastructure["database_status"]["errors"].append("Database file does not exist")

        except Exception as e:
            infrastructure["database_status"]["errors"].append(f"Database check failed: {e}")

        # Check key directories with detailed analysis
        directories_to_check = [
            ("src", "Main source code directory"),
            ("tests", "Test suite directory"),
            ("src/data", "Data layer implementation"),
            ("src/agent", "Agent layer implementation"),
            ("src/process", "Process layer implementation"),
            ("src/endpoint", "Endpoint layer implementation"),
            ("src/benchmarking", "Benchmarking and evaluation"),
            ("src/stats", "Statistics collection"),
            ("src/config", "Configuration management"),
            ("data", "Database and data storage")
        ]

        for dir_path, description in directories_to_check:
            full_path = self.project_root / dir_path
            dir_info = {
                "exists": full_path.exists(),
                "is_directory": full_path.is_dir() if full_path.exists() else False,
                "description": description,
                "file_count": 0,
                "subdirectory_count": 0
            }

            if full_path.exists() and full_path.is_dir():
                try:
                    items = list(full_path.iterdir())
                    dir_info["file_count"] = len([item for item in items if item.is_file()])
                    dir_info["subdirectory_count"] = len([item for item in items if item.is_dir()])
                except Exception as e:
                    dir_info["error"] = str(e)

            infrastructure["directory_status"][dir_path] = dir_info

        # Check critical Python packages
        critical_packages = [
            "pytest", "pydantic", "yaml", "requests", "pathlib",
            "asyncio", "sqlite3", "json", "subprocess"
        ]

        for package in critical_packages:
            try:
                if package == "sqlite3":
                    import sqlite3
                    infrastructure["dependency_status"]["critical_packages"][package] = {
                        "available": True,
                        "version": sqlite3.sqlite_version,
                        "builtin": True
                    }
                elif package == "pathlib" or package == "asyncio" or package == "json" or package == "subprocess":
                    # Built-in modules
                    infrastructure["dependency_status"]["critical_packages"][package] = {
                        "available": True,
                        "builtin": True
                    }
                else:
                    import importlib
                    module = importlib.import_module(package)
                    version = getattr(module, '__version__', 'unknown')
                    infrastructure["dependency_status"]["critical_packages"][package] = {
                        "available": True,
                        "version": version,
                        "builtin": False
                    }
            except ImportError:
                infrastructure["dependency_status"]["missing_packages"].append(package)
                infrastructure["dependency_status"]["critical_packages"][package] = {
                    "available": False,
                    "builtin": False
                }
            except Exception as e:
                infrastructure["dependency_status"]["warnings"].append(f"Error checking {package}: {e}")

        return infrastructure

    def get_implementation_state(self) -> Dict[str, Any]:
        """Get current implementation state with comprehensive analysis"""
        print("  Assessing implementation state...")

        impl_state = {
            "core_components": {},
            "api_implementation": {
                "real_api_calls": 0,
                "simulation_indicators": 0,
                "uses_real_apis": False,
                "api_quality_score": 0,
                "analysis_errors": []
            },
            "module_import_status": {},
            "implementation_health": {
                "overall_score": 0,
                "critical_issues": [],
                "warnings": [],
                "recommendations": []
            }
        }

        # Check for key implementation files with detailed analysis
        key_files = {
            'claim_system': ('src/data/models.py', 'Core claim data models and types'),
            'prompt_system': ('src/agent/prompt_system.py', 'Enhanced prompt system with reasoning'),
            'real_api_integration': ('src/benchmarking/real_api_claim_system.py', 'Real API integration layer'),
            'improved_claim_system': ('src/benchmarking/improved_claim_system.py', 'Improved claim processing'),
            'enhanced_evaluation': ('src/benchmarking/enhanced_local_evaluation.py', 'Enhanced evaluation methods'),
            'configuration_models': ('src/config/settings_models.py', 'Configuration management'),
            'sqlite_manager': ('src/data/optimized_sqlite_manager.py', 'Database operations'),
            'llm_processor': ('src/process/llm_processor.py', 'LLM processing layer')
        }

        for component, (file_path, description) in key_files.items():
            full_path = self.project_root / file_path
            component_info = {
                'exists': full_path.exists(),
                'file_path': file_path,
                'description': description,
                'size_bytes': 0,
                'line_count': 0,
                'class_count': 0,
                'function_count': 0,
                'last_modified': None,
                'analysis_error': None
            }

            if full_path.exists():
                try:
                    # File statistics
                    stat = full_path.stat()
                    component_info['size_bytes'] = stat.st_size
                    component_info['last_modified'] = datetime.fromtimestamp(stat.st_mtime).isoformat()

                    # Code analysis
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    lines = content.split('\n')
                    component_info['line_count'] = len(lines)

                    # Count classes and functions
                    import re
                    component_info['class_count'] = len(re.findall(r'^class\s+\w+', content, re.MULTILINE))
                    component_info['function_count'] = len(re.findall(r'^def\s+\w+', content, re.MULTILINE))

                except Exception as e:
                    component_info['analysis_error'] = str(e)

            impl_state['core_components'][component] = component_info

        # Analyze API implementation quality
        real_api_file = self.project_root / "src" / "benchmarking" / "real_api_claim_system.py"
        api_files = [
            ("src/benchmarking/real_api_claim_system.py", "Primary API integration"),
            ("src/process/llm_processor.py", "LLM processing"),
            ("src/endpoint/api_endpoints.py", "API endpoints")
        ]

        total_real_indicators = 0
        total_simulation_indicators = 0

        for api_file_path, description in api_files:
            full_path = self.project_root / api_file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Enhanced indicators for real API calls
                    real_api_indicators = [
                        "requests.post", "requests.get", "requests.put", "requests.delete",
                        "api.z.ai", "openrouter.ai", "api.openai.com", "generativelanguage.googleapis.com",
                        "timeout=", "headers=", "Authorization:", "Bearer ", "api_key",
                        "httpx.", "aiohttp.", "urllib.", "fetch(", "async def.*api"
                    ]

                    # Enhanced indicators for simulation
                    simulation_indicators = [
                        "mock", "Mock", "simulate", "simulation", "hardcoded", "placeholder",
                        "fake_", "dummy_", "test_response", "mock_response", "return \"mock\"",
                        "# This is a simulation", "TODO: replace with real API"
                    ]

                    real_count = sum(content.count(indicator) for indicator in real_api_indicators)
                    simulation_count = sum(content.count(indicator) for indicator in simulation_indicators)

                    total_real_indicators += real_count
                    total_simulation_indicators += simulation_count

                    impl_state['api_implementation']['analysis_errors'].append(
                        f"{api_file_path}: {real_count} real indicators, {simulation_count} simulation indicators"
                    )

                except Exception as e:
                    impl_state['api_implementation']['analysis_errors'].append(
                        f"Error analyzing {api_file_path}: {e}"
                    )

        impl_state['api_implementation']['real_api_calls'] = total_real_indicators
        impl_state['api_implementation']['simulation_indicators'] = total_simulation_indicators
        impl_state['api_implementation']['uses_real_apis'] = total_real_indicators > total_simulation_indicators

        # Calculate API quality score
        if total_real_indicators + total_simulation_indicators > 0:
            impl_state['api_implementation']['api_quality_score'] = round(
                (total_real_indicators / (total_real_indicators + total_simulation_indicators)) * 100, 1
            )

        # Test critical module imports
        critical_modules = [
            ('src.data.models', 'Claim'),
            ('src.agent.prompt_system', 'PromptSystem'),
            ('src.config.settings_models', 'ConjectureSettings'),
            ('src.data.optimized_sqlite_manager', 'SQLiteManager')
        ]

        for module_path, class_name in critical_modules:
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    module_path.replace('.', '/'),
                    self.project_root / f"{module_path.replace('.', '/')}.py"
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    impl_state['module_import_status'][module_path] = {
                        'importable': True,
                        'class_available': hasattr(module, class_name) if module else False,
                        'error': None
                    }
                else:
                    impl_state['module_import_status'][module_path] = {
                        'importable': False,
                        'class_available': False,
                        'error': 'Module spec not found'
                    }
            except Exception as e:
                impl_state['module_import_status'][module_path] = {
                    'importable': False,
                    'class_available': False,
                    'error': str(e)
                }

        # Calculate implementation health score
        core_components_present = sum(1 for comp in impl_state['core_components'].values() if comp['exists'])
        total_core_components = len(impl_state['core_components'])
        modules_importable = sum(1 for status in impl_state['module_import_status'].values() if status['importable'])
        total_modules = len(impl_state['module_import_status'])

        # Health score calculation
        component_score = (core_components_present / total_core_components) * 50 if total_core_components > 0 else 0
        import_score = (modules_importable / total_modules) * 30 if total_modules > 0 else 0
        api_score = impl_state['api_implementation']['api_quality_score'] * 0.2

        impl_state['implementation_health']['overall_score'] = round(component_score + import_score + api_score, 1)

        # Generate recommendations
        if impl_state['implementation_health']['overall_score'] < 50:
            impl_state['implementation_health']['critical_issues'].append("Low implementation health score")
        if not impl_state['api_implementation']['uses_real_apis']:
            impl_state['implementation_health']['warnings'].append("API implementation appears to be mostly simulated")
        if modules_importable < total_modules:
            impl_state['implementation_health']['warnings'].append("Some critical modules cannot be imported")

        return impl_state

    def save_stats(self, filename: str = "STATS.yaml") -> str:
        """Save statistics to YAML file"""
        output_path = self.project_root / filename

        try:
            import yaml
        except ImportError:
            # Fallback to JSON if PyYAML not available
            output_path = self.project_root / filename.replace('.yaml', '.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, indent=2, default=str)
            print(f"Saved as JSON to: {output_path}")
            return str(output_path)

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.stats, f, default_flow_style=False, indent=2, allow_unicode=True)

        print(f"Statistics saved to: {output_path}")
        return str(output_path)

def main():
    """Main statistics generation function"""
    start_time = time.time()

    generator = ProjectStatsGenerator()
    stats = generator.generate_all_stats()

    # Add generation metadata (ensure metadata exists)
    if 'metadata' not in stats:
        stats['metadata'] = {}

    stats['metadata']['generation_time_seconds'] = round(time.time() - start_time, 2)
    stats['metadata']['generator_version'] = "1.0.0"
    stats['metadata']['platform'] = sys.platform

    # Save statistics
    output_file = generator.save_stats()

    print(f"\n[SUCCESS] Statistics generation complete!")
    print(f"[TIME] Generation time: {stats['metadata']['generation_time_seconds']:.2f}s")
    print(f"[OUTPUT] File saved to: {output_file}")

    return stats

if __name__ == "__main__":
    main()