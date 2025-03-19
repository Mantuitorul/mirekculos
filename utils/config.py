#!/usr/bin/env python3
# utils/config.py
"""
Configuration management utilities for the pipeline.
Handles environment variables, logging setup, and directory management.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv, find_dotenv

def setup_logging(level=logging.INFO):
    """
    Set up logging configuration for the application.
    
    Args:
        level: Logging level (default: INFO)
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("moviepy").setLevel(logging.WARNING)

def load_environment() -> Dict[str, Any]:
    """
    Load environment variables from .env file.
    
    Returns:
        Dict with API keys
    
    Raises:
        ValueError: If required API keys are missing
    """
    logger = logging.getLogger(__name__)
    
    env_file = find_dotenv()
    if env_file:
        logger.info(f"Found .env file at: {env_file}")
    else:
        logger.warning("No .env file found! Make sure to provide API keys directly.")
    
    load_dotenv(env_file, override=True)
    
    # Load all HeyGen API keys (looking for HEYGEN_API_KEY_BE, HEYGEN_API_KEY_MIR, etc.)
    heygen_api_keys = []
    for key in os.environ:
        if key.startswith("HEYGEN_API_KEY_"):
            heygen_api_keys.append(os.environ[key])
    
    # Fallback to single key if no prefixed keys found
    if not heygen_api_keys:
        heygen_api_key = os.getenv("HEYGEN_API_KEY")
        if heygen_api_key:
            heygen_api_keys.append(heygen_api_key)
    
    elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pexels_api_key = os.getenv("PEXELS_API_KEY")
    
    if not heygen_api_keys:
        raise ValueError("No HEYGEN_API_KEY found in environment variables")
    
    if not elevenlabs_api_key:
        raise ValueError("ELEVENLABS_API_KEY not found in environment variables")
    
    if not pexels_api_key:
        logger.warning("PEXELS_API_KEY not found in environment variables. B-roll fetching will not work.")
    
    if not openai_api_key:
        logger.warning("OPENAI_API_KEY not found in environment variables. ChatGPT integration will not work.")
    
    logger.info(f"Loaded {len(heygen_api_keys)} HeyGen API keys")
        
    return {
        "HEYGEN_API_KEYS": heygen_api_keys,
        "ELEVENLABS_API_KEY": elevenlabs_api_key,
        "OPENAI_API_KEY": openai_api_key,
        "PEXELS_API_KEY": pexels_api_key
    }

def ensure_output_dir(directory: Path) -> Path:
    """
    Ensure the output directory exists, creating it if necessary.
    
    Args:
        directory: Path to the output directory
        
    Returns:
        The created/existing directory path
    """
    directory.mkdir(exist_ok=True, parents=True)
    return directory

# Initialize package structure
def init_package_structure():
    """Initialize the package directory structure if it doesn't exist."""
    # Create required directories
    required_dirs = [
        Path("audio"),
        Path("video"),
        Path("utils"),
        Path("output")
    ]
    
    for directory in required_dirs:
        directory.mkdir(exist_ok=True)
        init_file = directory / "__init__.py"
        if not init_file.exists():
            init_file.touch()
    
    # Create root __init__.py
    root_init = Path("__init__.py")
    if not root_init.exists():
        root_init.touch()

# Initialize package structure when module is imported
init_package_structure()