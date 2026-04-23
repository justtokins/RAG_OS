import json
import os

# Load configuration files
def load_config(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

# Load all configs
general_settings = load_config('general_settings.json')
agent_settings = load_config('agent_settings.json')
function_calls = load_config('function_calls.json')