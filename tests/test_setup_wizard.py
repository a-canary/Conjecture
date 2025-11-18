"""
Comprehensive tests for the Setup Wizard

Tests the simplified configuration system to ensure it covers
all the essential functionality while maintaining simplicity.
"""

import os
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
import pytest

# Import the wizard
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config.setup_wizard import SetupWizard, SimpleProvider, quick_setup, check_status, auto_setup_ollama

class TestSetupWizard:
    """Test the SetupWizard class"""
    
    def setup_method(self):
        """Setup test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        self.wizard = SetupWizard(str(self.test_path))
        
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_quick_status_no_env_file(self):
        """Test quick status with no .env file"""
        status = self.wizard.quick_status()
        
        assert not status['configured']
        assert status['provider'] is None
        assert not status['env_file_exists']
    
    def test_quick_status_empty_env_file(self):
        """Test quick status with empty .env file"""
        (self.test_path / '.env').write_text('')
        
        status = self.wizard.quick_status()
        
        assert not status['configured']
        assert status['provider'] is None
        assert status['env_file_exists']
    
    def test_quick_status_incomplete_config(self):
        """Test quick status with incomplete configuration"""
        (self.test_path / '.env').write_text(
            'Conjecture_LLM_PROVIDER=openai\n'
            'OPENAI_API_KEY=your_key_here\n'
        )
        
        status = self.wizard.quick_status()
        
        assert not status['configured']
        assert status['missing_api_key']
        assert status['provider'] == 'Openai'
    
    def test_quick_status_configured_ollama(self):
        """Test quick status with properly configured Ollama"""
        (self.test_path / '.env').write_text(
            'Conjecture_LLM_PROVIDER=ollama\n'
            'Conjecture_LLM_API_URL=http://localhost:11434\n'
            'Conjecture_LLM_MODEL=llama2\n'
        )
        
        status = self.wizard.quick_status()
        
        assert status['configured']
        assert status['provider'] == 'Ollama'
        assert status['provider_type'] == 'local'
        assert status['model'] == 'llama2'
        assert status['api_url'] == 'http://localhost:11434'
    
    def test_quick_status_configured_openai(self):
        """Test quick status with properly configured OpenAI"""
        (self.test_path / '.env').write_text(
            'Conjecture_LLM_PROVIDER=openai\n'
            'Conjecture_LLM_API_URL=https://api.openai.com/v1\n'
            'Conjecture_LLM_MODEL=gpt-3.5-turbo\n'
            'OPENAI_API_KEY=sk-test1234567890abcdef1234567890abcdef12345678\n'
        )
        
        status = self.wizard.quick_status()
        
        assert status['configured']
        assert status['provider'] == 'Openai'
        assert status['provider_type'] == 'cloud'
        assert status['model'] == 'gpt-3.5-turbo'
    
    def test_auto_detect_local_with_ollama(self):
        """Test local service detection with mocked Ollama"""
        with patch('config.setup_wizard.requests') as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_requests.get.return_value = mock_response
            
            detected = self.wizard.auto_detect_local()
            
            assert 'ollama' in detected
            mock_requests.get.assert_any_call('http://localhost:11434', timeout=3)
    
    def test_auto_detect_local_no_services(self):
        """Test local service detection with no services"""
        with patch('config.setup_wizard.requests') as mock_requests:
            mock_requests.get.side_effect = Exception("Connection failed")
            
            with patch('builtins.print') as mock_print:
                detected = self.wizard.auto_detect_local()
                
                assert detected == []
    
    def test_update_env_file_create_new(self):
        """Test creating new .env file"""
        config = {
            'Conjecture_LLM_PROVIDER': 'ollama',
            'Conjecture_LLM_API_URL': 'http://localhost:11434',
            'Conjecture_LLM_MODEL': 'llama2'
        }
        
        result = self.wizard.update_env_file(config)
        
        assert result
        assert (self.test_path / '.env').exists()
        
        content = (self.test_path / '.env').read_text()
        assert 'Conjecture_LLM_PROVIDER=ollama' in content
        assert 'Conjecture_EMBEDDING_MODEL=all-MiniLM-L6-v2' in content  # Default added
    
    def test_update_env_file_existing(self):
        """Test updating existing .env file"""
        (self.test_path / '.env').write_text(
            '# Existing config\n'
            'Conjecture_DB_PATH=custom.db\n'
        )
        
        config = {
            'Conjecture_LLM_PROVIDER': 'openai',
            'Conjecture_LLM_API_URL': 'https://api.openai.com/v1',
            'OPENAI_API_KEY': 'sk-test123'
        }
        
        result = self.wizard.update_env_file(config)
        
        assert result
        
        content = (self.test_path / '.env').read_text()
        assert 'Conjecture_LLM_PROVIDER=openai' in content
        assert 'Conjecture_DB_PATH=custom.db' in content  # Preserved
        
        # Check backup was created
        backups = list(self.test_path.glob('.env.backup.*'))
        assert len(backups) == 1
    
    def test_update_env_file_with_api_key(self):
        """Test updating .env file with API key (should be masked)"""
        config = {
            'OPENAI_API_KEY': 'sk-test1234567890abcdef1234567890abcdef12345678',
            'Conjecture_LLM_PROVIDER': 'openai'
        }
        
        result = self.wizard.update_env_file(config)
        
        assert result
        
        content = (self.test_path / '.env').read_text()
        assert 'sk-test1234567890abcdef...345678' in content  # Masked in comment
        assert os.path.getchmod((self.test_path / '.env')) & 0o777 == 0o600  # Secure permissions
    
    def test_validate_api_key_format_openai(self):
        """Test OpenAI API key format validation"""
        valid_key = 'sk-proj-abcd1234567890efghijklmnopqrstuvwxyz1234567890'
        invalid_key = 'invalid-key'
        
        assert self.wizard._validate_api_key_format('openai', valid_key)
        # Invalid format but should still return False for this specific pattern
        assert not self.wizard._validate_api_key_format('openai', invalid_key)
    
    def test_validate_api_key_format_unknown_provider(self):
        """Test API key validation for unknown provider"""
        # Should allow any format for unknown providers
        assert self.wizard._validate_api_key_format('unknown', 'any-key')
    
    def test_mask_api_key(self):
        """Test API key masking"""
        short_key = 'test'
        long_key = 'sk-proj-abcd1234567890efghijklmnopqrstuvwxyz1234567890'
        
        masked = self.wizard._mask_api_key(short_key)
        assert len(masked) == 4
        assert '*' in masked
        
        masked = self.wizard._mask_api_key(long_key)
        assert masked.startswith('sk-p')
        assert masked.endswith('5678')
        assert '*' in masked
        assert len(masked) == len(long_key)
    
    def test_test_endpoint_success(self):
        """Test successful endpoint test"""
        with patch('config.setup_wizard.requests') as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_requests.get.return_value = mock_response
            
            result = self.wizard._test_endpoint('http://localhost:11434')
            
            assert result
            mock_requests.get.assert_called_once_with('http://localhost:11434', timeout=3)
    
    def test_test_endpoint_failure(self):
        """Test failed endpoint test"""
        with patch('config.setup_wizard.requests') as mock_requests:
            mock_requests.get.side_effect = Exception("Connection failed")
            
            result = self.wizard._test_endpoint('http://localhost:11434')
            
            assert not result
    
    def test_read_env_file_not_exists(self):
        """Test reading non-existent .env file"""
        result = self.wizard._read_env_file()
        assert result == {}
    
    def test_read_env_file_valid(self):
        """Test reading valid .env file"""
        (self.test_path / '.env').write_text(
            '# Comment\n'
            'Conjecture_LLM_PROVIDER=ollama\n'
            'Conjecture_MODEL=llama2\n'
            '\n'
            '  OPENAI_API_KEY=sk-test123  \n'
        )
        
        result = self.wizard._read_env_file()
        
        expected = {
            'Conjecture_LLM_PROVIDER': 'ollama',
            'Conjecture_MODEL': 'llama2',
            'OPENAI_API_KEY': 'sk-test123'
        }
        
        assert result == expected
    
    def test_create_env_example(self):
        """Test creating .env.example file"""
        self.wizard._create_env_example()
        
        assert (self.test_path / '.env.example').exists()
        content = (self.test_path / '.env.example').read_text()
        assert 'Conjecture Environment Variables Template' in content
        assert 'Conjecture_LLM_PROVIDER=ollama' in content
    
    def test_create_backup(self):
        """Test creating .env backup"""
        (self.test_path / '.env').write_text('test content')
        
        backup = self.wizard._create_backup()
        
        assert backup is not None
        assert backup.exists()
        assert backup.name.startswith('.env.backup.')
        assert backup.read_text() == 'test content'


class TestInteractiveFeatures:
    """Test interactive features with mocked input"""
    
    def setup_method(self):
        """Setup test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        self.wizard = SetupWizard(str(self.test_path))
    
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('builtins.input')
    @patch('config.setup_wizard.requests')
    @patch('builtins.print')
    def test_interactive_setup_ollama(self, mock_print, mock_requests, mock_input):
        """Test interactive setup for Ollama"""
        # Mock successful Ollama detection
        mock_response = Mock()
        mock_response.status_code = 200
        mock_requests.get.return_value = mock_response
        
        # Mock user input
        mock_input.side_effect = [
            '1',  # Choose Ollama
            '',   # Default model
            '',   # Default API URL
            'y'   # Confirm and save
        ]
        
        with patch.object(self.wizard, '_test_endpoint', return_value=True):
            result = self.wizard.interactive_setup()
        
        assert result
        assert (self.test_path / '.env').exists()
        
        content = (self.test_path / '.env').read_text()
        assert 'Conjecture_LLM_PROVIDER=ollama' in content
    
    @patch('builtins.input')
    @patch('builtins.print')
    def test_interactive_setup_openai(self, mock_print, mock_input):
        """Test interactive setup for OpenAI"""
        # Mock user input
        mock_input.side_effect = [
            '3',  # Choose OpenAI (assuming local providers not detected)
            'sk-proj-abcd1234567890efghijklmnopqrstuvwxyz1234567890',  # API key
            '',   # Default model
            '',   # Default API URL
            'y'   # Confirm and save
        ]
        
        result = self.wizard.interactive_setup()
        
        assert result
        assert (self.test_path / '.env').exists()
        
        content = (self.test_path / '.env').read_text()
        assert 'Conjecture_LLM_PROVIDER=openai' in content
        assert 'OPENAI_API_KEY=' in content
    
    @patch('builtins.input')
    def test_interactive_setup_cancelled(self, mock_input):
        """Test cancelled interactive setup"""
        mock_input.side_effect = KeyboardInterrupt()
        
        result = self.wizard.interactive_setup()
        
        assert not result


class TestConvenienceFunctions:
    """Test the convenience functions"""
    
    def setup_method(self):
        """Setup test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
    
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_check_status_not_configured(self):
        """Test check status when not configured"""
        status = check_status(str(self.test_path))
        
        assert not status['configured']
        assert not status['env_file_exists']
    
    def test_check_status_configured(self):
        """Test check status when configured"""
        (self.test_path / '.env').write_text(
            'Conjecture_LLM_PROVIDER=ollama\n'
            'Conjecture_LLM_API_URL=http://localhost:11434\n'
            'Conjecture_LLM_MODEL=llama2\n'
        )
        
        status = check_status(str(self.test_path))
        
        assert status['configured']
        assert status['provider'] == 'Ollama'
    
    @patch('config.setup_wizard.SetupWizard.interactive_setup')
    def test_quick_setup_not_configured(self, mock_setup):
        """Test quick setup when not configured"""
        mock_setup.return_value = True
        
        result = quick_setup(str(self.test_path))
        
        assert result
        mock_setup.assert_called_once()
    
    @patch('builtins.print')
    def test_quick_setup_already_configured(self, mock_print):
        """Test quick setup when already configured"""
        (self.test_path / '.env').write_text(
            'Conjecture_LLM_PROVIDER=ollama\n'
            'Conjecture_LLM_API_URL=http://localhost:11434\n'
            'Conjecture_LLM_MODEL=llama2\n'
        )
        
        result = quick_setup(str(self.test_path))
        
        assert result
        mock_print.assert_any_call('✅ Already configured with Ollama (llama2)')
    
    @patch('config.setup_wizard.SetupWizard.auto_detect_local')
    @patch('config.setup_wizard.SetupWizard.update_env_file')
    def test_auto_setup_ollama_detected(self, mock_update, mock_detect):
        """Test auto Ollama setup when detected"""
        mock_detect.return_value = ['ollama']
        mock_update.return_value = True
        
        result = auto_setup_ollama(str(self.test_path))
        
        assert result
        mock_update.assert_called_once()
    
    @patch('config.setup_wizard.SetupWizard.auto_detect_local')
    def test_auto_setup_ollama_not_detected(self, mock_detect):
        """Test auto Ollama setup when not detected"""
        mock_detect.return_value = []
        
        with patch('builtins.print') as mock_print:
            result = auto_setup_ollama(str(self.test_path))
            
            assert not result
            mock_print.assert_any_call('❌ Ollama not available or setup failed')
    
    @patch('config.setup_wizard.SetupWizard.quick_status')
    def test_auto_setup_ollama_already_configured(self, mock_status):
        """Test auto Ollama setup when already configured"""
        mock_status.return_value = {'configured': True}
        
        with patch('builtins.print') as mock_print:
            result = auto_setup_ollama(str(self.test_path))
            
            assert result
            mock_print.assert_any_call('✅ Already configured')


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def setup_method(self):
        """Setup test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        self.wizard = SetupWizard(str(self.test_path))
    
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_invalid_env_file_handling(self):
        """Test handling of malformed .env file"""
        (self.test_path / '.env').write_text('invalid content without equals signs')
        
        # Should not crash
        status = self.wizard.quick_status()
        assert not status['configured']
        
        # Should still be able to update
        config = {'Conjecture_LLM_PROVIDER': 'ollama'}
        result = self.wizard.update_env_file(config)
        assert result
    
    def test_permission_denied_env_file(self):
        """Test handling when .env file cannot be written"""
        # Create a directory where .env should be
        (self.test_path / '.env').mkdir()
        
        config = {'Conjecture_LLM_PROVIDER': 'ollama'}
        result = self.wizard.update_env_file(config)
        
        assert not result
    
    def test_empty_config_update(self):
        """Test updating with empty config"""
        result = self.wizard.update_env_file({})
        
        assert result
        assert (self.test_path / '.env').exists()
        
        # Should have defaults
        content = (self.test_path / '.env').read_text()
        assert 'Conjecture_EMBEDDING_MODEL=all-MiniLM-L6-v2' in content
    
    def test_unknown_provider_config(self):
        """Test configuration with unknown provider"""
        (self.test_path / '.env').write_text(
            'Conjecture_LLM_PROVIDER=unknown_provider\n'
        )
        
        status = self.wizard.quick_status()
        
        assert not status['configured']
        assert status['provider'] is None


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])