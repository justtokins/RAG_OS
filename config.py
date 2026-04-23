import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Load configuration files relative to this module.
def load_config(file_name):
    file_path = BASE_DIR / file_name
    if file_path.exists():
        with file_path.open('r', encoding='utf-8') as f:
            return json.load(f)
    raise FileNotFoundError(f"Configuration file not found: {file_path}")

# Load all configs
general_settings = load_config('general_settings.json')
agent_settings = load_config('agent_settings.json')
function_calls = load_config('function_calls.json')