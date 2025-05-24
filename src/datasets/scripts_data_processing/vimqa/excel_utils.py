import pandas as pd

def save_to_excel(data, output_path):
    """
    Save data to Excel file.
    
    Args:
        data: List of dictionaries containing the data
        output_path: Path to save the Excel file
    """
    df = pd.DataFrame(data)
    df.to_excel(output_path, index=False)
    print(f"Created Excel file: {output_path}")

def create_excel_files(corpus_data, qa_data, contexts_data, output_dir, input_base, suffix=""):
    """
    Create Excel files for all data types.
    
    Args:
        corpus_data: Corpus data to save
        qa_data: QA pairs data to save
        contexts_data: Contexts gold data to save
        output_dir: Directory to save output files
        input_base: Base name for the files
        suffix: Optional suffix to add to filenames
    """
    # Save corpus to Excel
    save_to_excel(corpus_data, f"{output_dir}/corpus_{input_base}{suffix}.xlsx")
    
    # Save QA pairs to Excel
    save_to_excel(qa_data, f"{output_dir}/qa_pairs_{input_base}{suffix}.xlsx")
    
    # Save contexts gold to Excel
    save_to_excel(contexts_data, f"{output_dir}/contexts_gold_{input_base}{suffix}.xlsx") 