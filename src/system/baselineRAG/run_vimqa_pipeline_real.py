"""
Run RAG Pipeline cho VIMQA_dev collection
Đọc questions từ Excel, chạy qua RAG pipeline, và ghi kết quả ra file mới
"""

import pandas as pd
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

# Setup paths
current_file = Path(__file__)
sys.path.append(str(current_file.parent))

# Create outputs directory if it doesn't exist
outputs_dir = current_file.parent / "outputs"
outputs_dir.mkdir(exist_ok=True)

# Import RAG Pipeline
from pipelineRAG import RAGPipeline

class VIMQARunner:
    """
    Class để chạy RAG Pipeline trên VIMQA dataset
    """
    
    def __init__(
        self,
        collection_name: str = "VIMQA_dev",
        retriever_type: str = "vector",  # vector hoặc hybrid
        k: int = 5,
        generator_model: str = "gpt-4o-mini",
        temperature: float = 0.1
    ):
        """
        Khởi tạo VIMQA Runner
        
        Args:
            collection_name: Tên collection trong Qdrant
            retriever_type: Loại retriever (vector, bm25, hybrid)
            k: Số documents retrieve
            generator_model: Model OpenAI
            temperature: Temperature cho generation
        """
        self.collection_name = collection_name
        self.retriever_type = retriever_type
        self.k = k
        self.generator_model = generator_model
        self.temperature = temperature
        
        # Initialize RAG Pipeline
        print(f"🚀 Initializing VIMQA Pipeline...")
        print(f"   Collection: {collection_name}")
        print(f"   Retriever: {retriever_type}")
        print(f"   K documents: {k}")
        print(f"   Model: {generator_model}")
        
        self.rag_pipeline = RAGPipeline(
            retriever_type=retriever_type,
            vector_store_type="qdrant",
            documents=None,  # Sử dụng existing collection
            collection_name=collection_name,
            k=k,
            generator_model=generator_model,
            temperature=temperature
        )
        
        print(f"✅ Pipeline initialized successfully!")
    
    def load_questions_from_excel(self, file_path: str, question_column: str = "question", nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Load questions từ file Excel
        
        Args:
            file_path: Đường dẫn file Excel
            question_column: Tên cột chứa câu hỏi
            nrows: Số rows muốn đọc (None = đọc tất cả)
            
        Returns:
            DataFrame chứa questions
        """
        try:
            print(f"📖 Loading questions from: {file_path}")
            
            # Đọc file Excel
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                df = pd.read_excel(file_path, nrows=nrows)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path, nrows=nrows)
            else:
                raise ValueError("Unsupported file format. Use .xlsx, .xls, or .csv")
            
            print(f"✅ Loaded {len(df)} rows")
            print(f"📋 Columns: {list(df.columns)}")
            
            # Kiểm tra column question có tồn tại không
            if question_column not in df.columns:
                print(f"⚠️  Column '{question_column}' not found. Available columns: {list(df.columns)}")
                # Tự động detect question column
                possible_cols = [col for col in df.columns if 'question' in col.lower() or 'câu hỏi' in col.lower() or 'query' in col.lower()]
                if possible_cols:
                    question_column = possible_cols[0]
                    print(f"✅ Auto-detected question column: '{question_column}'")
                else:
                    raise ValueError(f"Cannot find question column. Available: {list(df.columns)}")
            
            # Filter out empty questions
            original_len = len(df)
            df = df[df[question_column].notna() & (df[question_column] != "")]
            print(f"📝 Valid questions: {len(df)}/{original_len}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading Excel file: {e}")
            raise
    
    def process_questions(
        self, 
        df: pd.DataFrame, 
        question_column: str = "question",
        batch_size: int = 10,
        save_intermediate: bool = True
    ) -> pd.DataFrame:
        """
        Xử lý tất cả questions qua RAG pipeline
        
        Args:
            df: DataFrame chứa questions
            question_column: Tên cột question
            batch_size: Số questions xử lý cùng lúc
            save_intermediate: Có save kết quả trung gian không
            
        Returns:
            DataFrame với kết quả
        """
        results = []
        total_questions = len(df)
        
        print(f"\n🤖 Processing {total_questions} questions...")
        print(f"📦 Batch size: {batch_size}")
        
        for i in range(0, total_questions, batch_size):
            batch_end = min(i + batch_size, total_questions)
            batch_df = df.iloc[i:batch_end].copy()
            
            print(f"\n📦 Processing batch {i//batch_size + 1}/{(total_questions-1)//batch_size + 1}")
            print(f"   Questions {i+1}-{batch_end}/{total_questions}")
            
            batch_results = []
            
            for idx, row in batch_df.iterrows():
                question = row[question_column]
                
                try:
                    print(f"   🔍 Q{idx+1}: {question[:50]}...")
                    
                    # Query through RAG pipeline
                    result = self.rag_pipeline.query(
                        question=question,
                        return_sources=True,
                        return_documents=False
                    )
                    
                    # Prepare result row
                    result_row = row.copy()
                    result_row['ai_answer'] = result.get('answer', 'Error: No answer generated')
                    result_row['sources'] = str(result.get('sources', []))
                    result_row['num_documents_used'] = result.get('num_documents_used', 0)
                    result_row['has_error'] = 'error' in result
                    result_row['error_message'] = result.get('error', '')
                    
                    batch_results.append(result_row)
                    print(f"   ✅ Completed")
                    
                except Exception as e:
                    print(f"   ❌ Error: {e}")
                    result_row = row.copy()
                    result_row['ai_answer'] = f"Error: {str(e)}"
                    result_row['sources'] = '[]'
                    result_row['num_documents_used'] = 0
                    result_row['has_error'] = True
                    result_row['error_message'] = str(e)
                    batch_results.append(result_row)
            
            results.extend(batch_results)
            
            # Save intermediate results
            if save_intermediate and len(results) > 0:
                temp_df = pd.DataFrame(results)
                temp_file = outputs_dir / f"temp_results_batch_{i//batch_size + 1}.xlsx"
                temp_df.to_excel(temp_file, index=False)
                print(f"💾 Saved intermediate results to: {temp_file}")
        
        return pd.DataFrame(results)
    
    def save_results(
        self, 
        results_df: pd.DataFrame, 
        output_file: str = None,
        include_metadata: bool = True
    ) -> str:
        """
        Lưu kết quả ra file Excel
        
        Args:
            results_df: DataFrame chứa kết quả
            output_file: Tên file output
            include_metadata: Có bao gồm metadata không
            
        Returns:
            Đường dẫn file đã lưu
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"vimqa_results_{timestamp}.xlsx"
        
        # Ensure output file is in outputs directory
        output_path = outputs_dir / output_file
        
        try:
            print(f"\n💾 Saving results to: {output_path}")
            
            if include_metadata:
                # Add metadata sheet
                with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                    # Main results
                    results_df.to_excel(writer, sheet_name='Results', index=False)
                    
                    # Metadata
                    metadata = {
                        'run_timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                        'collection_name': [self.collection_name],
                        'retriever_type': [self.retriever_type],
                        'k_documents': [self.k],
                        'generator_model': [self.generator_model],
                        'temperature': [self.temperature],
                        'total_questions': [len(results_df)],
                        'successful_answers': [len(results_df[~results_df['has_error']])],
                        'failed_answers': [len(results_df[results_df['has_error']])]
                    }
                    metadata_df = pd.DataFrame(metadata)
                    metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
                    
                    print(f"✅ Saved with metadata")
            else:
                results_df.to_excel(output_path, index=False)
                print(f"✅ Saved results only")
            
            return str(output_path)
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            raise
    
    def run_full_pipeline(
        self,
        input_file: str,
        output_file: str = None,
        question_column: str = "question",
        batch_size: int = 10,
        nrows: Optional[int] = None
    ) -> str:
        """
        Chạy full pipeline: load → process → save
        
        Args:
            input_file: File Excel input
            output_file: File Excel output
            question_column: Cột chứa questions
            batch_size: Batch size
            nrows: Số rows muốn đọc (None = đọc tất cả)
            
        Returns:
            Đường dẫn file kết quả
        """
        print(f"\n🚀 Starting VIMQA Full Pipeline")
        print(f"{'='*50}")
        
        # Step 1: Load questions
        df = self.load_questions_from_excel(input_file, question_column, nrows=nrows)
        
        # Step 2: Process questions
        results_df = self.process_questions(
            df, 
            question_column=question_column,
            batch_size=batch_size
        )
        
        # Step 3: Save results
        output_path = self.save_results(results_df, output_file)
        
        # Step 4: Summary
        total = len(results_df)
        successful = len(results_df[~results_df['has_error']])
        failed = len(results_df[results_df['has_error']])
        
        print(f"\n🎯 Pipeline Summary:")
        print(f"   Total questions: {total}")
        print(f"   Successful: {successful} ({successful/total*100:.1f}%)")
        print(f"   Failed: {failed} ({failed/total*100:.1f}%)")
        print(f"   Output file: {output_path}")
        
        return output_path

def main():
    """
    Main function để chạy VIMQA pipeline
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Run VIMQA RAG Pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input Excel file path")
    parser.add_argument("--output", "-o", help="Output Excel file path")
    parser.add_argument("--question_column", "-q", default="question", help="Question column name")
    parser.add_argument("--collection", "-c", default="VIMQA_dev", help="Qdrant collection name")
    parser.add_argument("--retriever", "-r", default="vector", choices=["vector", "bm25", "hybrid"], help="Retriever type")
    parser.add_argument("--k", type=int, default=5, help="Number of documents to retrieve")
    parser.add_argument("--batch_size", "-b", type=int, default=10, help="Batch size for processing")
    parser.add_argument("--model", "-m", default="gpt-4o-mini", help="OpenAI model")
    parser.add_argument("--temperature", "-t", type=float, default=0.1, help="Generation temperature")
    parser.add_argument("--nrows", "-n", type=int, help="Number of rows to process (default: all)")
    
    args = parser.parse_args()
    
    try:
        # Initialize runner
        runner = VIMQARunner(
            collection_name=args.collection,
            retriever_type=args.retriever,
            k=args.k,
            generator_model=args.model,
            temperature=args.temperature
        )
        
        # Run pipeline
        output_file = runner.run_full_pipeline(
            input_file=args.input,
            output_file=args.output,
            question_column=args.question_column,
            batch_size=args.batch_size,
            nrows=args.nrows
        )
        
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"📄 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"\n💥 Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Example usage when run directly
    if len(sys.argv) == 1:
        print("🚀 VIMQA RAG Pipeline Runner")
        print("="*40)
        print("\nExample usage:")
        print("python run_vimqa_pipeline.py --input questions.xlsx --output results.xlsx")
        print("\nArguments:")
        print("  --input, -i      : Input Excel file with questions")
        print("  --output, -o     : Output Excel file (optional)")
        print("  --question_column: Column name containing questions (default: 'question')")
        print("  --collection, -c : Qdrant collection name (default: 'VIMQA_dev')")
        print("  --retriever, -r  : Retriever type (vector/bm25/hybrid, default: vector)")
        print("  --k              : Number of documents to retrieve (default: 5)")
        print("  --batch_size, -b : Batch size (default: 10)")
        print("  --model, -m      : OpenAI model (default: gpt-4o-mini)")
        print("  --temperature, -t: Generation temperature (default: 0.1)")
        print("  --nrows, -n      : Number of rows to process (default: all)")
        
        # Quick test với VIMQA dataset
        print("\n🧪 Running test with VIMQA dataset...")
        
        # Đường dẫn đến file input
        input_file = current_file.parent.parent.parent / "datasets" / "dataset_full" / "vimqa_processed" / "qa_pairs_vimqa_dev_300.xlsx"
        
        if not input_file.exists():
            print(f"❌ Input file not found: {input_file}")
            sys.exit(1)
            
        print(f"📖 Reading from: {input_file}")
        
        try:
            # Khởi tạo runner
            runner = VIMQARunner(
                collection_name="VIMQA_dev",
                retriever_type="vector",
                k=3,
                generator_model="gpt-4o-mini",
                temperature=0.1
            )
            
            # Chạy pipeline với 5 rows đầu
            output_file = runner.run_full_pipeline(
                input_file=str(input_file),
                question_column="question",
                batch_size=5,  # Process all 5 rows in one batch
                nrows=5  # Only read first 5 rows
            )
            
            print(f"✅ Test completed! Check: {output_file}")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            
    else:
        main()
