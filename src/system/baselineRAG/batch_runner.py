"""
Batch Runner cho VIMQA experiments
"""

import os
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Setup paths
current_file = Path(__file__)
sys.path.append(str(current_file.parent))

from system.baselineRAG.run_vimqa_pipeline_sample import VIMQARunner

def run_experiments(
    input_file: str,
    experiments: list,
    base_output_dir: str = "experiments"
):
    """
    Chạy nhiều experiments với các config khác nhau
    
    Args:
        input_file: File Excel input
        experiments: List các config experiments
        base_output_dir: Thư mục output
    """
    
    # Create output directory
    os.makedirs(base_output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_summary = []
    
    for i, exp_config in enumerate(experiments, 1):
        print(f"\n🧪 Running Experiment {i}/{len(experiments)}")
        print(f"{'='*50}")
        print(f"Config: {exp_config}")
        
        try:
            # Initialize runner with experiment config
            runner = VIMQARunner(**exp_config)
            
            # Create output filename
            exp_name = f"exp_{i}_{exp_config['retriever_type']}_k{exp_config['k']}"
            output_file = os.path.join(base_output_dir, f"{exp_name}_{timestamp}.xlsx")
            
            # Run pipeline
            result_file = runner.run_full_pipeline(
                input_file=input_file,
                output_file=output_file,
                batch_size=5
            )
            
            # Load results for summary
            results_df = pd.read_excel(result_file, sheet_name='Results')
            metadata_df = pd.read_excel(result_file, sheet_name='Metadata')
            
            # Add to summary
            summary_row = {
                'experiment': exp_name,
                'retriever_type': exp_config['retriever_type'],
                'k': exp_config['k'],
                'model': exp_config['generator_model'],
                'total_questions': len(results_df),
                'successful': len(results_df[~results_df['has_error']]),
                'success_rate': len(results_df[~results_df['has_error']]) / len(results_df) * 100,
                'output_file': result_file
            }
            results_summary.append(summary_row)
            
            print(f"✅ Experiment {i} completed successfully!")
            
        except Exception as e:
            print(f"❌ Experiment {i} failed: {e}")
            summary_row = {
                'experiment': f"exp_{i}_FAILED",
                'retriever_type': exp_config.get('retriever_type', 'unknown'),
                'k': exp_config.get('k', 'unknown'),
                'model': exp_config.get('generator_model', 'unknown'),
                'total_questions': 0,
                'successful': 0,
                'success_rate': 0,
                'output_file': f"FAILED: {str(e)}"
            }
            results_summary.append(summary_row)
    
    # Save summary
    summary_df = pd.DataFrame(results_summary)
    summary_file = os.path.join(base_output_dir, f"experiments_summary_{timestamp}.xlsx")
    summary_df.to_excel(summary_file, index=False)
    
    print(f"\n📊 Experiments Summary:")
    print(summary_df.to_string(index=False))
    print(f"\n💾 Summary saved to: {summary_file}")
    
    return summary_file

if __name__ == "__main__":
    # Example experiments
    experiments = [
        {
            'collection_name': 'VIMQA_dev',
            'retriever_type': 'vector',
            'k': 3,
            'generator_model': 'gpt-4o-mini',
            'temperature': 0.1
        },
        {
            'collection_name': 'VIMQA_dev',
            'retriever_type': 'vector',
            'k': 5,
            'generator_model': 'gpt-4o-mini',
            'temperature': 0.1
        },
        {
            'collection_name': 'VIMQA_dev',
            'retriever_type': 'hybrid',
            'k': 5,
            'generator_model': 'gpt-4o-mini',
            'temperature': 0.1
        }
    ]
    
    # Run experiments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        run_experiments(input_file, experiments)
    else:
        print("Usage: python batch_runner.py <input_excel_file>")
