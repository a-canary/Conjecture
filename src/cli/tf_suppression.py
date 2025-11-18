"""
TensorFlow warning suppression utilities.
Reduces console noise from TensorFlow deprecation warnings and informational messages.
"""
import os
import warnings
import logging
from typing import Optional


def suppress_tensorflow_warnings() -> bool:
    """
    Suppress TensorFlow deprecation warnings and informational messages.
    
    Returns:
        bool: True if suppression was successful, False otherwise
    """
    success = True
    
    try:
        # Environment variables for TensorFlow
        tf_env_vars = {
            'TF_ENABLE_ONEDNN_OPTS': '0',
            'TF_CPP_MIN_LOG_LEVEL': '2',
            'TF_DEPRECATION_WARNINGS': '0',
            'TF_WARNINGS': '0'
        }
        
        for var, value in tf_env_vars.items():
            os.environ[var] = value
        
        # General warning suppression
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
        
        # TensorFlow specific logging
        try:
            import tensorflow as tf
            
            # Set TensorFlow logging level to ERROR only
            tf.get_logger().setLevel('ERROR')
            
            # Disable TensorFlow v2 behavior warnings
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
            
            # Disable experimental features warnings
            if hasattr(tf, 'autograph'):
                tf.autograph.set_verbosity(0)
            
        except ImportError:
            # TensorFlow not installed, nothing to suppress
            pass
        
        # Keras specific warnings
        try:
            from tensorflow import keras
            # Suppress Keras warnings
            os.environ['KERAS_BACKEND'] = 'tensorflow'
        except ImportError:
            pass
        
        # Suppress other ML library warnings
        try:
            import torch
            # Suppress PyTorch warnings
            warnings.filterwarnings('ignore', category=UserWarning, module='torch')
        except ImportError:
            pass
        
    except Exception:
        success = False
    
    return success


def setup_ml_logging(level: str = 'ERROR') -> bool:
    """
    Setup consistent logging for machine learning libraries.
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        
    Returns:
        bool: True if setup was successful, False otherwise
    """
    success = True
    
    try:
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.ERROR),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            force=True  # Override existing configuration
        )
        
        # Suppress specific loggers
        suppressed_loggers = [
            'tensorflow',
            'keras',
            'absl',
            'matplotlib',
            'PIL',
            'urllib3.connectionpool'
        ]
        
        for logger_name in suppressed_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.ERROR)
            logger.propagate = False
        
    except Exception:
        success = False
    
    return success


def get_ml_environment_info() -> dict:
    """
    Get information about the ML environment and library versions.
    
    Returns:
        dict: Dictionary containing ML library information
    """
    info = {}
    
    # TensorFlow
    try:
        import tensorflow as tf
        info['tensorflow'] = {
            'version': tf.__version__,
            'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0,
            'built_with_cuda': tf.test.is_built_with_cuda()
        }
    except ImportError:
        info['tensorflow'] = None
    
    # PyTorch
    try:
        import torch
        info['pytorch'] = {
            'version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'mps_available': torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        }
    except ImportError:
        info['pytorch'] = None
    
    # Keras
    try:
        import keras
        info['keras'] = {'version': keras.__version__}
    except ImportError:
        info['keras'] = None
    
    # Scikit-learn
    try:
        import sklearn
        info['sklearn'] = {'version': sklearn.__version__}
    except ImportError:
        info['sklearn'] = None
    
    return info


def print_ml_environment_info(console=None) -> None:
    """
    Print ML environment information in a formatted way.
    
    Args:
        console: Console instance to use for printing
    """
    if console is None:
        from .encoding_handler import get_safe_console
        console = get_safe_console()
    
    info = get_ml_environment_info()
    
    console.print("🧠 Machine Learning Environment:", style="bold blue")
    
    for lib_name, lib_info in info.items():
        if lib_info:
            console.print(f"  • {lib_name.title()}: {lib_info['version']}", style="green")
            if lib_name == 'tensorflow' and lib_info.get('gpu_available'):
                console.print("    └─ GPU acceleration available", style="cyan")
            elif lib_name == 'pytorch' and lib_info.get('cuda_available'):
                console.print("    └─ CUDA acceleration available", style="cyan")
        else:
            console.print(f"  • {lib_name.title()}: Not installed", style="dim")


# Initialize TensorFlow warning suppression when module is imported
suppress_tensorflow_warnings()
setup_ml_logging()