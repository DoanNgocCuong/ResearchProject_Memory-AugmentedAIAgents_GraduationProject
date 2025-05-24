"""
Module đánh giá hiệu quả của khâu generation trong hệ thống RAG từ file Excel.
Đọc file Excel có 3 cột: question - answer - ai_answer và đánh giá kết quả.
"""

import pandas as pd
from typing import List, Dict, Any
from langchain_core.documents import Document
from layers._06_evaluation.evaluator_generation import GenerationEvaluator
import os
from pathlib import Path
import argparse

class ExcelGenerationEvaluator:
    """
    Class đánh giá hiệu quả của khâu generation từ file Excel.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0
    ):
        """
        Khởi tạo ExcelGenerationEvaluator.
        
        Args:
            model_name: Tên model AI sử dụng
            temperature: Độ sáng tạo (0.0 - 1.0)
        """
        self.evaluator = GenerationEvaluator(
            model_name=model_name,
            temperature=temperature
        )
    
    def evaluate_excel(
        self,
        input_file: str,
        output_file: str,
        context_column: str = "answer"
    ) -> None:
        """
        Đánh giá các câu trả lời từ file Excel và lưu kết quả.
        
        Args:
            input_file: Đường dẫn file Excel đầu vào
            output_file: Đường dẫn file Excel đầu ra
            context_column: Tên cột chứa context (mặc định là "answer")
        """
        # Đọc file Excel
        df = pd.read_excel(input_file)
        
        # Kiểm tra các cột cần thiết
        required_columns = ["question", "answer", "ai_answer"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Tạo list để lưu kết quả đánh giá
        evaluation_results = []
        
        # Đánh giá từng dòng
        for idx, row in df.iterrows():
            print(f"Evaluating row {idx + 1}/{len(df)}...")
            
            # Tạo document từ context
            context_doc = Document(
                page_content=row[context_column],
                metadata={"source": f"row_{idx}"}
            )
            
            # Đánh giá câu trả lời
            result = self.evaluator.evaluate_answer(
                question=row["question"],
                answer=row["ai_answer"],
                documents=[context_doc],
                reference_answer=row["answer"]
            )
            
            # Lưu kết quả
            evaluation_results.append({
                "question": row["question"],
                "reference_answer": row["answer"],
                "generated_answer": row["ai_answer"],
                "llm_score": result["llm_score"],
                "bleu_1": result["bleu_1"],
                "bleu_2": result["bleu_2"],
                "bleu_3": result["bleu_3"],
                "bleu_4": result["bleu_4"],
                "rouge_l": result["rouge_l"],
                "f1": result["f1"],
                "accuracy_score": result["detailed_scores"]["accuracy"],
                "completeness_score": result["detailed_scores"]["completeness"],
                "relevance_score": result["detailed_scores"]["relevance"],
                "clarity_score": result["detailed_scores"]["clarity"],
                "consistency_score": result["detailed_scores"]["consistency"],
                "missing_information": "\n".join(result["missing_information"]),
                "hallucinations": "\n".join(result["hallucinations"]),
                "quality_assessment": result["answer_quality"]["assessment"]
            })
        
        # Tạo DataFrame từ kết quả
        result_df = pd.DataFrame(evaluation_results)
        
        # Tính toán các metrics tổng hợp
        summary = {
            "Average LLM Score": result_df["llm_score"].mean(),
            "Average BLEU-1": result_df["bleu_1"].mean(),
            "Average BLEU-2": result_df["bleu_2"].mean(),
            "Average BLEU-3": result_df["bleu_3"].mean(),
            "Average BLEU-4": result_df["bleu_4"].mean(),
            "Average Rouge-L": result_df["rouge_l"].mean(),
            "Average F1": result_df["f1"].mean(),
            "Average Accuracy": result_df["accuracy_score"].mean(),
            "Average Completeness": result_df["completeness_score"].mean(),
            "Average Relevance": result_df["relevance_score"].mean(),
            "Average Clarity": result_df["clarity_score"].mean(),
            "Average Consistency": result_df["consistency_score"].mean()
        }
        
        # Tạo DataFrame cho summary
        summary_df = pd.DataFrame([summary])
        
        # Lưu kết quả vào file Excel
        with pd.ExcelWriter(output_file) as writer:
            result_df.to_excel(writer, sheet_name="Detailed Results", index=False)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
        
        print(f"\nEvaluation completed. Results saved to {output_file}")
        print("\nSummary of metrics:")
        for metric, value in summary.items():
            print(f"{metric}: {value:.4f}")

def main():
    """Run ExcelGenerationEvaluator with command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate RAG generation results from Excel file")
    parser.add_argument("--input", "-i", required=True, help="Input Excel file path (must contain question, answer, and ai_answer columns)")
    parser.add_argument("--output", "-o", help="Output Excel file path (default: evaluation_results.xlsx)")
    parser.add_argument("--model", "-m", default="gpt-4o-mini", help="Model name for evaluation (default: gpt-4o-mini)")
    parser.add_argument("--temperature", "-t", type=float, default=0.0, help="Temperature for evaluation (default: 0.0)")
    
    args = parser.parse_args()
    
    # Set default output file if not provided
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"evaluation_results_{input_path.stem}.xlsx")
    
    try:
        # Khởi tạo evaluator
        evaluator = ExcelGenerationEvaluator(
            model_name=args.model,
            temperature=args.temperature
        )
        
        # Đánh giá file Excel
        evaluator.evaluate_excel(
            input_file=args.input,
            output_file=args.output
        )
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
