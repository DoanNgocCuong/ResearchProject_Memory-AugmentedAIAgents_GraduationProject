import json
from typing import List, Dict, Any
import os

def save_jsonl(data: List[Dict[str, Any]], output_file: str):
    """
    Save data to JSONL file.
    
    Args:
        data: List of dictionaries to save
        output_file: Path to output file
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def load_json(input_file: str) -> List[Dict[str, Any]]:
    """
    Load JSON file.
    
    Args:
        input_file: Path to input file
        
    Returns:
        List of dictionaries from JSON file
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def convert_to_vietnamese_readable(input_file: str, output_file: str):
    """
    Convert JSON file to Vietnamese readable format (no unicode escape).
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file (readable)
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2) 