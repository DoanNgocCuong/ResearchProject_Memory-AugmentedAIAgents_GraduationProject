import json
from typing import Dict, List, Any
import argparse

def convert_hotpotqa_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a single HotpotQA example to the desired format.
    
    Args:
        example: A dictionary containing a HotpotQA example
        
    Returns:
        A dictionary in the new format
    """
    # Get list of supporting titles
    supporting_titles = {title for title, _ in example["supporting_facts"]}
    
    # Create new contexts list
    new_contexts = []
    for idx, (title, sentences) in enumerate(example["context"]):
        paragraph_text = " ".join(sentences)
        is_supporting = title in supporting_titles
        new_contexts.append({
            "idx": idx,
            "title": title,
            "paragraph_text": paragraph_text,
            "is_supporting": is_supporting
        })
    
    # Return new format dictionary
    return {
        "question_id": example["_id"],
        "question": example["question"],
        "contexts": new_contexts,
        "answer": example["answer"]
    }

def process_hotpotqa_file(input_file: str, output_file: str):
    """
    Process HotpotQA JSON file and save to JSONL format.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSONL file
    """
    # Read input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process each example and write to JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in data:
            converted = convert_hotpotqa_example(example)
            f.write(json.dumps(converted) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Convert HotpotQA JSON to JSONL format')
    parser.add_argument('input_file', help='Path to input JSON file')
    parser.add_argument('output_file', help='Path to output JSONL file')
    
    args = parser.parse_args()
    
    try:
        process_hotpotqa_file(args.input_file, args.output_file)
        print(f"Successfully converted {args.input_file} to {args.output_file}")
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main() 