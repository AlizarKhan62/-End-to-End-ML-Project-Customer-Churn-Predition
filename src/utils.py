
import logging
import json
from pathlib import Path
from datetime import datetime

def setup_logging(log_file: str = 'logs/app.log'):
    """Setup logging configuration"""
    Path('logs').mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def save_json(data: dict, filepath: str):
    """Save dictionary to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(filepath: str) -> dict:
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def get_timestamp() -> str:
    """Get current timestamp"""
    return datetime.now().isoformat()