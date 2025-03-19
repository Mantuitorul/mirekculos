#!/usr/bin/env python3
"""
Configuration management for the video generation pipeline.
Handles environment variables, logging setup, and directory management.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv, find_dotenv

class Config:
    """Configuration manager for the video pipeline."""
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            env_file: Path to .env file (default: auto-detect)
        """
        self.logger = logging.getLogger(__name__)
        self._config = {}
        self.setup_logging()
        self.load_environment(env_file)
    
    def setup_logging(self, level=logging.INFO):
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
    
    def load_environment(self, env_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Load environment variables from .env file.
        
        Args:
            env_file: Path to .env file (default: auto-detect)
            
        Returns:
            Dict with API keys
        
        Raises:
            ValueError: If required API keys are missing
        """
        if env_file:
            env_path = env_file
        else:
            env_path = find_dotenv()
            
        if env_path:
            self.logger.info(f"Found .env file at: {env_path}")
            load_dotenv(env_path, override=True)
        else:
            self.logger.warning("No .env file found! Make sure to provide API keys directly.")
        
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
        
        self._config = {
            "heygen_api_keys": heygen_api_keys,
            "elevenlabs_api_key": elevenlabs_api_key,
            "openai_api_key": openai_api_key,
            "pexels_api_key": pexels_api_key
        }
        
        # Log loaded keys count
        self.logger.info(f"Loaded {len(heygen_api_keys)} HeyGen API keys")
        
        # Log warnings for missing optional keys
        if not elevenlabs_api_key:
            self.logger.warning("ELEVENLABS_API_KEY not found in environment variables")
        
        if not pexels_api_key:
            self.logger.warning("PEXELS_API_KEY not found in environment variables. B-roll fetching will not work.")
        
        if not openai_api_key:
            self.logger.warning("OPENAI_API_KEY not found in environment variables. ChatGPT integration will not work.")
            
        return self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self._config[key] = value
    
    @property
    def heygen_api_keys(self) -> List[str]:
        """Get HeyGen API keys."""
        return self._config.get("heygen_api_keys", [])
    
    @property
    def elevenlabs_api_key(self) -> Optional[str]:
        """Get ElevenLabs API key."""
        return self._config.get("elevenlabs_api_key")
    
    @property
    def openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key."""
        return self._config.get("openai_api_key")
    
    @property
    def pexels_api_key(self) -> Optional[str]:
        """Get Pexels API key."""
        return self._config.get("pexels_api_key")


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


def load_environment(env_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Load environment variables from .env file (legacy function).
    
    Args:
        env_file: Path to .env file (default: auto-detect)
        
    Returns:
        Dict with API keys
    """
    config = Config(env_file)
    return {
        "heygen_api_keys": config.heygen_api_keys,
        "elevenlabs_api_key": config.elevenlabs_api_key,
        "openai_api_key": config.openai_api_key,
        "pexels_api_key": config.pexels_api_key
    } 