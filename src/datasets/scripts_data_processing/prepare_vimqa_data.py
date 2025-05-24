import argparse
from vimqa import (
    create_corpus,
    create_qa_pairs,
    create_contexts_gold,
    save_jsonl,
    load_json,
    convert_to_vietnamese_readable,
    create_excel_files
)
import os

def process_vimqa_file(input_file: str, output_dir: str, num_lines: int = None):
    """
    Process VIMQA JSON file and create all necessary output files.
    
    Args:
        input_file: Path to input JSON file
        output_dir: Directory to save output files
        num_lines: Number of lines to process (None for all lines)
    """
    # Read input JSON file
    data = load_json(input_file)
    
    # Limit the number of lines if specified
    if num_lines is not None:
        data = data[:num_lines]
        print(f"Processing {num_lines} lines from input file")
    
    input_base = os.path.splitext(os.path.basename(input_file))[0]
    
    # Add number of lines to filename if specified
    suffix = f"_{num_lines}" if num_lines is not None else ""
    
    # Create data
    corpus_data = create_corpus(data)
    qa_data = create_qa_pairs(data)
    contexts_data = create_contexts_gold(data)
    
    # Save JSONL files
    save_jsonl(corpus_data, f"{output_dir}/corpus_{input_base}{suffix}.jsonl")
    save_jsonl(qa_data, f"{output_dir}/qa_pairs_{input_base}{suffix}.jsonl")
    save_jsonl(contexts_data, f"{output_dir}/contexts_gold_{input_base}{suffix}.jsonl")
    
    # Save Excel files
    create_excel_files(
        corpus_data=corpus_data,
        qa_data=qa_data,
        contexts_data=contexts_data,
        output_dir=output_dir,
        input_base=input_base,
        suffix=suffix
    )

    # Convert to Vietnamese readable JSON
    vi_json_path = os.path.join(output_dir, f"{input_base}{suffix}_vi.json")
    convert_to_vietnamese_readable(input_file, vi_json_path)
    print(f"Created Vietnamese readable file: {vi_json_path}")

def main():
    parser = argparse.ArgumentParser(description='Prepare VIMQA data for retrieval-based QA')
    parser.add_argument('input_file', help='Path to input JSON file')
    parser.add_argument('output_dir', help='Directory to save output files')
    parser.add_argument('-n', '--num_lines', type=int, help='Number of lines to process (optional)')
    
    args = parser.parse_args()
    
    try:
        process_vimqa_file(args.input_file, args.output_dir, args.num_lines)
        print(f"Successfully processed {args.input_file}")
        print(f"Output files created in {args.output_dir}:")
        suffix = f"_{args.num_lines}" if args.num_lines is not None else ""
        print(f"- corpus_{args.input_file}{suffix}.jsonl and .xlsx")
        print(f"- qa_pairs_{args.input_file}{suffix}.jsonl and .xlsx")
        print(f"- contexts_gold_{args.input_file}{suffix}.jsonl and .xlsx")
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main() 