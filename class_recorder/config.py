"""Configuration management with environment variables"""
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    def __init__(self, config_path='config.yaml'):
        with open(config_path) as f:
            self._config = yaml.safe_load(f)
        
        # API Keys from environment
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
    
    def get(self, path, default=None):
        """Get nested config: config.get('recording.sample_rate')"""
        keys = path.split('.')
        value = self._config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
        return value if value is not None else default

# Global config instance
config = Config()
