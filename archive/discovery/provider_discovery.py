"""
Provider Discovery System

Main discovery engine that orchestrates:
- Service detection
- Configuration management  
- User interaction
- Automatic and manual discovery modes
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging

from .service_detector import ServiceDetector, DetectedProvider, discover_providers
from .config_updater import ConfigUpdater

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProviderDiscovery:
    """Main provider discovery system"""
    
    def __init__(self, project_root: Optional[str] = None, timeout: int = 3):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.timeout = timeout
        self.config_updater = ConfigUpdater(str(self.project_root))
        
    async def run_automatic_discovery(
        self, 
        auto_configure: bool = True,
        preferred_provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run automatic discovery and configuration"""
        
        result = {
            'success': False,
            'providers': [],
            'primary_provider': None,
            'configuration_updated': False,
            'message': '',
            'errors': []
        }
        
        try:
            # Step 1: Detect providers
            logger.info("Starting provider discovery...")
            providers = await self._discover_providers()
            
            if not providers:
                result['message'] = 'No LLM providers detected. Please install Ollama/LM Studio or set API keys.'
                return result
            
            result['providers'] = [self._provider_to_dict(p) for p in providers]
            logger.info(f"Detected {len(providers)} providers")
            
            # Step 2: Select primary provider
            primary_provider = self._select_primary_provider(providers, preferred_provider)
            result['primary_provider'] = self._provider_to_dict(primary_provider)
            
            # Step 3: Update configuration if requested
            if auto_configure:
                config_success, config_result = await self._update_configuration(
                    providers, primary_provider.name, auto_confirm=True
                )
                result['configuration_updated'] = config_success
                result['config_result'] = config_result
                
                if config_success:
                    result['message'] = f"Successfully configured {primary_provider.name} as primary provider"
                    result['success'] = True
                else:
                    result['message'] = "Provider detection succeeded but configuration failed"
                    result['errors'].append(str(config_result.get('error', 'Unknown configuration error')))
            else:
                result['success'] = True
                result['message'] = f"Providers detected successfully. Use auto_configure=True to update configuration."
            
        except Exception as e:
            logger.error(f"Discovery failed: {e}")
            result['errors'].append(str(e))
            result['message'] = f"Discovery failed: {e}"
        
        return result
    
    async def run_manual_discovery(self) -> Dict[str, Any]:
        """Run manual discovery with user interaction"""
        
        result = {
            'success': False,
            'providers': [],
            'selected_provider': None,
            'configuration_updated': False,
            'message': '',
            'interactive_mode': True
        }
        
        try:
            print("\n🔍 Conjecture Provider Discovery")
            print("=" * 40)
            
            # Step 1: Detect providers
            print("Scanning for LLM providers...")
            providers = await self._discover_providers()
            
            if not providers:
                print("\n❌ No LLM providers detected!")
                print("\nTo set up providers:")
                print("1. Install Ollama: https://ollama.ai/")
                print("2. Install LM Studio: https://lmstudio.ai/")  
                print("3. Set environment variables (see .env.example)")
                return result
            
            result['providers'] = [self._provider_to_dict(p) for p in providers]
            
            # Step 2: Display detected providers
            self._display_providers(providers)
            
            # Step 3: Interactive selection
            selected_provider = self._interactive_provider_selection(providers)
            if not selected_provider:
                result['message'] = "No provider selected"
                return result
            
            result['selected_provider'] = self._provider_to_dict(selected_provider)
            
            # Step 4: Confirm configuration
            if self._confirm_configuration(selected_provider):
                config_success, config_result = await self._update_configuration(
                    providers, selected_provider.name, auto_confirm=False
                )
                
                if config_success:
                    result['configuration_updated'] = True
                    result['success'] = True
                    result['message'] = f"Successfully configured {selected_provider.name}!"
                    print(f"\n✅ Configuration updated successfully!")
                    self._display_configuration_summary(config_result)
                else:
                    result['message'] = "Configuration failed"
                    print(f"\n❌ Configuration failed: {config_result.get('error', 'Unknown error')}")
            else:
                result['message'] = "Configuration cancelled by user"
                print("\n⚠️ Configuration cancelled")
        
        except KeyboardInterrupt:
            result['message'] = "Discovery cancelled by user"
            print("\n⚠️ Discovery cancelled")
        except Exception as e:
            logger.error(f"Manual discovery failed: {e}")
            result['errors'].append(str(e))
            result['message'] = f"Discovery failed: {e}"
        
        return result
    
    async def quick_check(self) -> Dict[str, Any]:
        """Quick check for available providers without configuration"""
        try:
            providers = await self._discover_providers()
            summary = self._get_discovery_summary(providers)
            return {
                'success': True,
                'providers': [self._provider_to_dict(p) for p in providers],
                'summary': summary
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'providers': [],
                'summary': {}
            }
    
    async def _discover_providers(self) -> List[DetectedProvider]:
        """Discover all available providers"""
        async with ServiceDetector(timeout=self.timeout) as detector:
            return await detector.detect_all()
    
    def _select_primary_provider(
        self, 
        providers: List[DetectedProvider], 
        preferred: Optional[str] = None
    ) -> DetectedProvider:
        """Select primary provider based on priority and preference"""
        if preferred:
            # Try to find preferred provider
            for provider in providers:
                if provider.name.lower() == preferred.lower():
                    return provider
        
        # Prefer local services
        local_providers = [p for p in providers if p.type == 'local']
        if local_providers:
            return local_providers[0]  # Already sorted by priority
        
        # Fall back to first available
        return providers[0]
    
    async def _update_configuration(
        self, 
        providers: List[DetectedProvider], 
        primary_provider: str,
        auto_confirm: bool = False
    ) -> Tuple[bool, Dict[str, Any]]:
        """Update configuration with detected providers"""
        return self.config_updater.update_config_with_providers(
            providers, primary_provider, auto_confirm
        )
    
    def _provider_to_dict(self, provider: DetectedProvider) -> Dict[str, Any]:
        """Convert provider to dictionary for serialization"""
        return {
            'name': provider.name,
            'type': provider.type,
            'endpoint': provider.endpoint,
            'api_key_env_var': provider.api_key_env_var,
            'models_count': len(provider.models),
            'models': provider.models[:5],  # Limit to first 5 models
            'priority': provider.priority,
            'status': provider.status,
            'info': provider.info
        }
    
    def _display_providers(self, providers: List[DetectedProvider]):
        """Display detected providers in a user-friendly format"""
        print(f"\n📋 Detected {len(providers)} LLM Provider(s):")
        print("-" * 40)
        
        for i, provider in enumerate(providers, 1):
            priority_symbol = "⭐" if provider.priority <= 2 else ""
            provider_type = "🏠 Local" if provider.type == 'local' else "☁️ Cloud"
            
            print(f"\n{i}. {provider.name} {priority_symbol}")
            print(f"   Type: {provider_type}")
            print(f"   Models: {len(provider.models)} available")
            
            if provider.endpoint:
                print(f"   Endpoint: {provider.endpoint}")
            
            if provider.api_key_env_var:
                masked_key = self.config_updater.mask_api_key(
                    os.getenv(provider.api_key_env_var, 'Not set')
                )
                print(f"   API Key: {masked_key} ({provider.api_key_env_var})")
            
            if provider.models:
                # Show first few models
                sample_models = provider.models[:3]
                models_text = ", ".join(sample_models)
                if len(provider.models) > 3:
                    models_text += f" (+{len(provider.models) - 3} more)"
                print(f"   Sample models: {models_text}")
    
    def _interactive_provider_selection(self, providers: List[DetectedProvider]) -> Optional[DetectedProvider]:
        """Interactive provider selection"""
        print(f"\n🎯 Select Primary Provider:")
        print("-" * 30)
        
        for i, provider in enumerate(providers, 1):
            priority_note = " (Recommended)" if provider.priority <= 2 else ""
            print(f"{i}. {provider.name}{priority_note}")
        
        while True:
            try:
                choice = input(f"\nEnter selection (1-{len(providers)}) or press Enter for default [1]: ").strip()
                
                if not choice:
                    choice = "1"
                
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(providers):
                        selected = providers[idx]
                        print(f"\n✅ Selected: {selected.name}")
                        return selected
                
                print("❌ Invalid selection. Please try again.")
                
            except (KeyboardInterrupt, EOFError):
                return None
    
    def _confirm_configuration(self, provider: DetectedProvider) -> bool:
        """Confirm configuration update with user"""
        print(f"\n⚙️ Configuration Preview:")
        print("-" * 30)
        print(f"Provider: {provider.name}")
        print(f"Type: {provider.type}")
        
        if provider.endpoint:
            print(f"Endpoint: {provider.endpoint}")
        
        if provider.models:
            print(f"Default Model: {provider.models[0]}")
        
        if provider.api_key_env_var:
            masked_key = self.config_updater.mask_api_key(
                os.getenv(provider.api_key_env_var, 'Not set')
            )
            print(f"API Key: {masked_key}")
        
        print("\nThis will update your .env file with the above configuration.")
        
        while True:
            try:
                confirm = input("\nProceed with configuration? [Y/n]: ").strip().lower()
                return confirm in ('', 'y', 'yes')
            except (KeyboardInterrupt, EOFError):
                return False
    
    def _display_configuration_summary(self, config_result: Dict[str, Any]):
        """Display configuration summary"""
        print("\n📄 Configuration Summary:")
        print("-" * 30)
        print(f"Primary Provider: {config_result.get('primary_provider', 'Unknown')}")
        print(f"Provider Type: {config_result.get('provider_type', 'Unknown')}")
        
        updates = config_result.get('updates', {})
        if updates:
            print("\nEnvironment Variables Updated:")
            for key, value in updates.items():
                print(f"  {key}: {value}")
        
        if config_result.get('backup_created'):
            print("\n💾 Backup of previous .env file created")
    
    def _get_discovery_summary(self, providers: List[DetectedProvider]) -> Dict[str, Any]:
        """Get discovery summary statistics"""
        detector = ServiceDetector()
        return detector.get_detected_summary(providers)
    
    def get_configuration_status(self) -> Dict[str, Any]:
        """Get current configuration status"""
        return self.config_updater.get_config_status()

# Convenience functions for easy access
async def discover_and_configure(
    auto_configure: bool = True,
    preferred_provider: Optional[str] = None,
    project_root: Optional[str] = None,
    timeout: int = 3
) -> Dict[str, Any]:
    """Convenience function for automatic discovery and configuration"""
    discovery = ProviderDiscovery(project_root, timeout)
    return await discovery.run_automatic_discovery(auto_configure, preferred_provider)

async def manual_discovery(project_root: Optional[str] = None, timeout: int = 3) -> Dict[str, Any]:
    """Convenience function for manual discovery"""
    discovery = ProviderDiscovery(project_root, timeout)
    return await discovery.run_manual_discovery()

async def quick_check_providers(project_root: Optional[str] = None, timeout: int = 3) -> Dict[str, Any]:
    """Convenience function for quick provider check"""
    discovery = ProviderDiscovery(project_root, timeout)
    return await discovery.quick_check()