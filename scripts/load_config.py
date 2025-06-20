import yaml
import os

def get_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config
    except Exception as e:
        print(f"Error loading config file: {e}")
        raise
