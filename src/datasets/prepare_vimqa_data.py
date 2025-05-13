import argparse
from vimqa import (
    create_corpus,
    create_qa_pairs,
    create_contexts_gold,
    save_jsonl,
    load_json,
    convert_to_vietnamese_readable
)
import os

def process_vimqa_file(input_file: str, output_dir: str):
    """
    Process VIMQA JSON file and create all necessary output files.
    
    Args:
        input_file: Path to input JSON file
        output_dir: Directory to save output files
    """
    # Read input JSON file
    data = load_json(input_file)
    input_base = os.path.splitext(os.path.basename(input_file))[0]
    
    # Create and save corpus
    save_jsonl(create_corpus(data), f"{output_dir}/corpus_{input_base}.jsonl")
    # Create and save QA pairs
    save_jsonl(create_qa_pairs(data), f"{output_dir}/qa_pairs_{input_base}.jsonl")
    # Create and save contexts gold
    save_jsonl(create_contexts_gold(data), f"{output_dir}/contexts_gold_{input_base}.jsonl")

    # Convert to Vietnamese readable JSON
    vi_json_path = os.path.join(output_dir, f"{input_base}_vi.json")
    convert_to_vietnamese_readable(input_file, vi_json_path)
    print(f"Created Vietnamese readable file: {vi_json_path}")

def main():
    parser = argparse.ArgumentParser(description='Prepare VIMQA data for retrieval-based QA')
    parser.add_argument('input_file', help='Path to input JSON file')
    parser.add_argument('output_dir', help='Directory to save output files')
    
    args = parser.parse_args()
    
    try:
        process_vimqa_file(args.input_file, args.output_dir)
        print(f"Successfully processed {args.input_file}")
        print(f"Output files created in {args.output_dir}:")
        print("- corpus_{input_base}.jsonl")
        print("- qa_pairs_{input_base}.jsonl")
        print("- contexts_gold_{input_base}.jsonl")
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main() 