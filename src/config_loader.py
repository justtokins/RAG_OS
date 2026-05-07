"""
Enhanced config loader with validation and schema enforcement.
"""
import json
from pathlib import Path
from typing import Any, Dict


class ConfigLoader:
    """Load and validate configuration files with schema checking."""
    
    BASE_DIR = Path(__file__).resolve().parent.parent / "config"
    
    # Define schema for each config file
    SCHEMAS = {
        'general_settings.json': {
            'required_keys': ['app', 'embedding', 'ingestion', 'llm', 'retrieval', 'persona'],
            'nested': {
                'app': ['name', 'version', 'log_level'],
                'embedding': ['model_name', 'vector_db_dir'],
                'ingestion': ['chunk_size', 'chunk_overlap', 'separators', 'uploads_dir'],
                'llm': ['provider', 'model', 'temperature', 'max_tokens'],
                'retrieval': ['top_k', 'history_limit'],
                'persona': ['name', 'role', 'tone', 'target_level'],
            }
        },
        'agent_settings.json': {
            'required_keys': ['intent_triggers', 'tools_enabled'],
            'nested': {
                'intent_triggers': ['definition', 'quiz', 'compare', 'wikipedia'],
            }
        },
        'function_calls.json': {
            'required_keys': ['tools'],
        }
    }
    
    def __init__(self):
        self.configs = {}
    
    def load_config(self, file_name: str) -> Dict[str, Any]:
        """Load a config file with validation."""
        file_path = self.BASE_DIR / file_name
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with file_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate schema if defined
        if file_name in self.SCHEMAS:
            self._validate_schema(file_name, data)
        
        self.configs[file_name] = data
        return data
    
    def _validate_schema(self, file_name: str, data: Dict[str, Any]) -> None:
        """Validate loaded config against schema."""
        schema = self.SCHEMAS[file_name]
        
        # Check required top-level keys
        for key in schema['required_keys']:
            if key not in data:
                raise ValueError(f"{file_name}: Missing required key '{key}'")
        
        # Check nested keys if defined
        if 'nested' in schema:
            for parent_key, child_keys in schema['nested'].items():
                if parent_key in data:
                    if not isinstance(data[parent_key], dict):
                        raise ValueError(f"{file_name}: '{parent_key}' must be a dict")
                    for child_key in child_keys:
                        if child_key not in data[parent_key]:
                            raise ValueError(
                                f"{file_name}: Missing required key '{parent_key}.{child_key}'"
                            )
    
    def load_all(self) -> tuple:
        """Load all config files and return tuple."""
        general = self.load_config('general_settings.json')
        agent = self.load_config('agent_settings.json')
        functions = self.load_config('function_calls.json')
        return general, agent, functions
    
    def get(self, file_name: str) -> Dict[str, Any]:
        """Get already-loaded config or raise error."""
        if file_name not in self.configs:
            return self.load_config(file_name)
        return self.configs[file_name]


# Initialize global loader
_loader = ConfigLoader()

# Load all configs on import
general_settings, agent_settings, function_calls = _loader.load_all()
