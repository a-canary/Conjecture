"""
System Diagnostics Module for Conjecture Config Wizard

Performs comprehensive system checks and provides detailed status reporting.
"""

import os
import sys
import platform
import subprocess
import requests
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import shutil
import psutil

@dataclass
class DiagnosticResult:
    """Result of a diagnostic check"""
    name: str
    status: str  # 'pass', 'warn', 'fail', 'info'
    message: str
    details: Optional[Dict[str, Any]] = None
    suggestion: Optional[str] = None

@dataclass
class SystemInfo:
    """System information summary"""
    platform: str
    python_version: str
    architecture: str
    memory_gb: float
    cpu_cores: int
    available_disk_gb: float

class SystemDiagnostics:
    """Comprehensive system diagnostics"""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.results: List[DiagnosticResult] = []
        
    def run_all_diagnostics(self) -> Dict[str, Any]:
        """Run complete diagnostic suite"""
        self.results = []
        
        # System information
        sys_info = self._get_system_info()
        
        # Core diagnostics
        self._check_python_version()
        self._check_dependencies()
        self._check_uv_availability()
        self._check_project_structure()
        self._check_configuration_files()
        self._check_local_providers()
        self._check_network_connectivity()
        self._check_disk_space()
        self._check_memory_usage()
        
        return {
            'system_info': asdict(sys_info),
            'results': [asdict(r) for r in self.results],
            'summary': self._generate_summary()
        }
    
    def _get_system_info(self) -> SystemInfo:
        """Collect basic system information"""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage(str(self.project_root))
        
        return SystemInfo(
            platform=platform.system(),
            python_version=sys.version,
            architecture=platform.machine(),
            memory_gb=round(memory.total / (1024**3), 2),
            cpu_cores=psutil.cpu_count(),
            available_disk_gb=round(disk.free / (1024**3), 2)
        )
    
    def _check_python_version(self) -> None:
        """Check Python version compatibility"""
        version = sys.version_info
        min_version = (3, 8)
        
        if version >= min_version:
            status = 'pass'
            message = f"Python {version.major}.{version.minor}.{version.micro} is compatible"
            suggestion = None
        else:
            status = 'fail'
            message = f"Python {version.major}.{version.minor}.{version.micro} is too old (requires 3.8+)"
            suggestion = "Please upgrade Python to version 3.8 or higher"
        
        self.results.append(DiagnosticResult(
            name="Python Version",
            status=status,
            message=message,
            details={'version': f"{version.major}.{version.minor}.{version.micro}", 'min_required': "3.8"},
            suggestion=suggestion
        ))
    
    def _check_dependencies(self) -> None:
        """Check if required dependencies are installed"""
        requirements_file = self.project_root / 'requirements.txt'
        
        if not requirements_file.exists():
            self.results.append(DiagnosticResult(
                name="Dependencies",
                status="warn",
                message="requirements.txt not found",
                suggestion="Run dependency installation"
            ))
            return
        
        # Read requirements
        try:
            requirements = requirements_file.read_text().strip().split('\n')
            missing = []
            installed = []
            
            for req in requirements:
                req = req.strip()
                if not req or req.startswith('#'):
                    continue
                
                # Extract package name (remove version specs)
                pkg_name = req.split('==')[0].split('>=')[0].split('<=')[0].split('~=')[0]
                
                try:
                    __import__(pkg_name.replace('-', '_'))
                    installed.append(req)
                except ImportError:
                    missing.append(req)
            
            if missing:
                status = 'fail'
                message = f"Missing {len(missing)} dependencies"
                suggestion = f"Run: uv pip install -r requirements.txt"
            else:
                status = 'pass'
                message = f"All {len(installed)} dependencies installed"
            
            self.results.append(DiagnosticResult(
                name="Dependencies",
                status=status,
                message=message,
                details={'installed': installed, 'missing': missing},
                suggestion=suggestion
            ))
            
        except Exception as e:
            self.results.append(DiagnosticResult(
                name="Dependencies",
                status="fail",
                message=f"Failed to check dependencies: {e}",
                suggestion="Check requirements.txt format"
            ))
    
    def _check_uv_availability(self) -> None:
        """Check if uv is available for dependency management"""
        try:
            result = subprocess.run(['uv', '--version'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version = result.stdout.strip()
                status = 'pass'
                message = f"uv {version} is available"
                suggestion = "Use 'uv pip install' for faster dependency installation"
            else:
                status = 'warn'
                message = "uv is not available"
                suggestion = "Install uv for faster dependency management: pip install uv"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            status = 'info'
            message = "uv is not installed (optional)"
            suggestion = "Install uv for modern Python package management: pip install uv"
        
        self.results.append(DiagnosticResult(
            name="UV Package Manager",
            status=status,
            message=message,
            suggestion=suggestion
        ))
    
    def _check_project_structure(self) -> None:
        """Check project structure integrity"""
        required_files = [
            'conjecture',
            'requirements.txt',
            '.env.example',
            'pyproject.toml'
        ]
        
        required_dirs = [
            'src',
            'data'
        ]
        
        missing_files = []
        missing_dirs = []
        
        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                missing_files.append(file_path)
        
        for dir_path in required_dirs:
            if not (self.project_root / dir_path).exists():
                missing_dirs.append(dir_path)
        
        if missing_files or missing_dirs:
            status = 'fail'
            message = f"Project structure incomplete"
            details = {'missing_files': missing_files, 'missing_dirs': missing_dirs}
            suggestion = "Ensure you're in the correct project directory"
        else:
            status = 'pass'
            message = "Project structure is complete"
            details = None
            suggestion = None
        
        self.results.append(DiagnosticResult(
            name="Project Structure",
            status=status,
            message=message,
            details=details,
            suggestion=suggestion
        ))
    
    def _check_configuration_files(self) -> None:
        """Check configuration status"""
        env_file = self.project_root / '.env'
        env_example = self.project_root / '.env.example'
        
        if env_file.exists():
            # Check if it's configured
            try:
                content = env_file.read_text()
                if 'your-api-key-here' in content or 'your_' in content:
                    status = 'warn'
                    message = ".env exists but needs configuration"
                    suggestion = "Run the setup wizard to configure your provider"
                else:
                    status = 'pass'
                    message = ".env is configured"
                    suggestion = None
            except Exception:
                status = 'fail'
                message = "Failed to read .env file"
                suggestion = "Check .env file permissions"
        else:
            status = 'info'
            message = ".env file not found"
            suggestion = "Copy .env.example to .env and configure"
        
        self.results.append(DiagnosticResult(
            name="Configuration",
            status=status,
            message=message,
            suggestion=suggestion
        ))
    
    def _check_local_providers(self) -> None:
        """Check for local LLM providers"""
        providers = {
            'Ollama': 'http://localhost:11434',
            'LM Studio': 'http://localhost:1234/v1'
        }
        
        available = []
        unavailable = []
        
        for name, endpoint in providers.items():
            try:
                response = requests.get(endpoint, timeout=2)
                if response.status_code == 200:
                    available.append(name)
                else:
                    unavailable.append(name)
            except:
                unavailable.append(name)
        
        if available:
            status = 'pass'
            message = f"Found {len(available)} local provider(s): {', '.join(available)}"
        else:
            status = 'info'
            message = "No local providers detected"
        
        self.results.append(DiagnosticResult(
            name="Local Providers",
            status=status,
            message=message,
            details={'available': available, 'unavailable': unavailable},
            suggestion="Install Ollama or LM Studio for local AI processing"
        ))
    
    def _check_network_connectivity(self) -> None:
        """Check basic network connectivity"""
        try:
            response = requests.get('https://httpbin.org/get', timeout=5)
            if response.status_code == 200:
                status = 'pass'
                message = "Network connectivity is working"
            else:
                status = 'warn'
                message = "Network connectivity issues detected"
        except requests.exceptions.Timeout:
            status = 'warn'
            message = "Network timeout (may be slow connection)"
        except requests.exceptions.ConnectionError:
            status = 'fail'
            message = "No network connectivity"
        except Exception as e:
            status = 'fail'
            message = f"Network check failed: {e}"
        
        network_suggestion = None
        if status == 'fail':
            network_suggestion = "Check internet connection for cloud providers"

        self.results.append(DiagnosticResult(
            name="Network Connectivity",
            status=status,
            message=message,
            suggestion=network_suggestion
        ))
    
    def _check_disk_space(self) -> None:
        """Check available disk space"""
        disk = psutil.disk_usage(str(self.project_root))
        free_gb = disk.free / (1024**3)
        
        if free_gb > 5:  # More than 5GB
            status = 'pass'
            message = f"Plenty of disk space: {free_gb:.1f}GB available"
        elif free_gb > 1:  # 1-5GB
            status = 'warn'
            message = f"Limited disk space: {free_gb:.1f}GB available"
            suggestion = "Consider freeing up disk space"
        else:  # Less than 1GB
            status = 'fail'
            message = f"Very low disk space: {free_gb:.1f}GB available"
            suggestion = "Free up disk space before proceeding"
        
        disk_suggestion = None
        if free_gb <= 5:
            disk_suggestion = "Consider freeing up disk space" if free_gb > 1 else "Free up disk space before proceeding"

        self.results.append(DiagnosticResult(
            name="Disk Space",
            status=status,
            message=message,
            details={'free_gb': round(free_gb, 2)},
            suggestion=disk_suggestion
        ))
    
    def _check_memory_usage(self) -> None:
        """Check available memory"""
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        if available_gb > 4:  # More than 4GB
            status = 'pass'
            message = f"Good memory availability: {available_gb:.1f}GB free"
        elif available_gb > 2:  # 2-4GB
            status = 'warn'
            message = f"Moderate memory: {available_gb:.1f}GB free"
            suggestion = "Close unnecessary applications for better performance"
        else:  # Less than 2GB
            status = 'fail'
            message = f"Low memory: {available_gb:.1f}GB free"
            suggestion = "Free up memory for optimal performance"
        
        memory_suggestion = None
        if available_gb <= 4:
            memory_suggestion = "Close unnecessary applications for better performance" if available_gb > 2 else "Free up memory for optimal performance"

        self.results.append(DiagnosticResult(
            name="Memory",
            status=status,
            message=message,
            details={'available_gb': round(available_gb, 2)},
            suggestion=memory_suggestion
        ))
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate diagnostic summary"""
        status_counts = {'pass': 0, 'warn': 0, 'fail': 0, 'info': 0}
        
        for result in self.results:
            status_counts[result.status] += 1
        
        # Overall status
        if status_counts['fail'] > 0:
            overall = 'fail'
            message = f"Critical issues found ({status_counts['fail']} failures)"
        elif status_counts['warn'] > 0:
            overall = 'warn'
            message = f"Some warnings ({status_counts['warn']} warnings)"
        else:
            overall = 'pass'
            message = "All systems ready"
        
        return {
            'overall_status': overall,
            'message': message,
            'status_counts': status_counts,
            'total_checks': len(self.results),
            'ready_for_setup': status_counts['fail'] == 0
        }

def run_diagnostics(project_root: Optional[Path] = None) -> Dict[str, Any]:
    """Convenience function to run diagnostics"""
    diagnostics = SystemDiagnostics(project_root)
    return diagnostics.run_all_diagnostics()