"""
Utility functions for generating various types of IDs
"""

import uuid
import random
import string
import time
from typing import Optional

def generate_session_id() -> str:
    """Generate a unique session ID"""
    return f"session_{uuid.uuid4().hex[:16]}_{int(time.time())}"

def generate_state_id() -> str:
    """Generate a unique state tracking ID"""
    return f"state_{uuid.uuid4().hex[:12]}_{int(time.time() * 1000) % 10000}"

def generate_workflow_id() -> str:
    """Generate a unique workflow execution ID"""
    return f"workflow_{uuid.uuid4().hex[:8]}_{int(time.time())}"

def generate_error_id() -> str:
    """Generate a unique error ID"""
    return f"error_{uuid.uuid4().hex[:8]}_{int(time.time())}"

def generate_template_id() -> str:
    """Generate a unique template ID"""
    return f"tpl_{uuid.uuid4().hex[:8]}_{random.choices(string.ascii_lowercase, k=3)[0]}"

def generate_context_id() -> str:
    """Generate a unique context ID"""
    return f"ctx_{uuid.uuid4().hex[:10]}_{int(time.time())}"

def generate_short_id(length: int = 8) -> str:
    """Generate a short random ID"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))