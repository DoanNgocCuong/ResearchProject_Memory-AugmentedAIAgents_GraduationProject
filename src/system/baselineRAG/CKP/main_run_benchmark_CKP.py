import os
import json
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
from pydantic import Field, BaseModel
import pandas as pd

from layers._01_data_ingestion.loader import load_faq_data, preprocess_faq_data
from layers._03_embedding.embedder import DocumentEmbedder
from layers._04_retrieval.retriever import DocumentRetriever
from layers._05_generation.generator import AnswerGenerator
from layers._06_evaluation.evaluator import RAGEvaluator

# Cài đặt thư viện cần thiết
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document, BaseRetriever
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import JSONLoader
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# System prompts
SYSTEM_PROMPT = """Bạn là một trợ lý AI chuyên nghiệp, có nhiệm vụ trả lời các câu hỏi về tính năng của ứng dụng.
Hãy tuân thủ các nguyên tắc sau:
1. Chỉ sử dụng thông tin từ ngữ cảnh được cung cấp
2. Trả lời ngắn gọn, rõ ràng và chính xác
3. Nếu không tìm thấy thông tin trong ngữ cảnh, hãy nói rằng bạn không biết
4. Sử dụng ngôn ngữ tự nhiên, thân thiện
5. Cung cấp thông tin chi tiết về các tính năng khi được hỏi"""

EVALUATION_PROMPT = """Bạn là một chuyên gia đánh giá hệ thống hỏi đáp.
Hãy đánh giá câu trả lời dựa trên các tiêu chí sau:
1. Độ chính xác (0-40 điểm): Thông tin có chính xác không?
2. Độ đầy đủ (0-30 điểm): Có bao quát đủ thông tin không?
3. Độ rõ ràng (0-20 điểm): Câu trả lời có dễ hiểu không?
4. Độ hữu ích (0-10 điểm): Câu trả lời có giúp ích cho người dùng không?

Tổng điểm: 0-100
Hãy đưa ra phản hồi chi tiết cho từng tiêu chí."""

@dataclass
class BenchmarkResult:
    """Lưu trữ kết quả benchmark cho một câu hỏi"""
    query: str
    expected_answer: str
    actual_answer: str
    evaluation_score: float
    evaluation_feedback: str
    relevant_docs: List[Document]
    processing_time: float

def load_benchmark_data(file_path: str) -> List[Dict[str, Any]]:
    """Đọc dữ liệu benchmark từ file JSON"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_vectordb(documents: List[Document], persist_directory: str = "data/chroma") -> FAISS:
    """Tạo vector database từ documents"""
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(documents=documents, embedding=embeddings)
    return vectordb

def run_benchmark(
    retriever: DocumentRetriever,
    generator: AnswerGenerator,
    evaluator: RAGEvaluator,
    benchmark_data: List[Dict[str, Any]]
) -> List[BenchmarkResult]:
    """Chạy benchmark và trả về kết quả"""
    results = []
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_results_{timestamp}.xlsx"
    
    # Tạo DataFrame rỗng
    df = pd.DataFrame(columns=[
        "query", "expected_answer", "actual_answer", 
        "response_time", "source_id", "timestamp",
        "evaluation_score", "evaluation_feedback"
    ])
    
    for i, item in enumerate(benchmark_data, 1):
        query = item["query"]
        expected_answer = item["expected_answer"]
        
        try:
            # Đo thời gian xử lý
            start_time = time.time()
            
            # Tìm kiếm tài liệu liên quan
            relevant_docs = retriever.retrieve_documents(query)
            
            # Tạo câu trả lời
            actual_answer = generator.generate_answer(query, relevant_docs)
            
            # Đánh giá câu trả lời
            evaluation = evaluator.evaluate_answer(query, actual_answer, relevant_docs)
            
            # Tính thời gian xử lý
            processing_time = time.time() - start_time
            
            # Lưu kết quả
            result = BenchmarkResult(
                query=query,
                expected_answer=expected_answer,
                actual_answer=actual_answer,
                evaluation_score=evaluation["score"],
                evaluation_feedback=evaluation["feedback"],
                relevant_docs=relevant_docs,
                processing_time=processing_time
            )
            results.append(result)
            
            # In tiến trình
            print(f"\nĐã xử lý câu hỏi {i}/{len(benchmark_data)}: {query}")
            print(f"Thời gian xử lý: {processing_time:.2f} giây")
            
            # Thêm kết quả vào DataFrame
            new_row = pd.DataFrame([{
                "query": query,
                "expected_answer": expected_answer,
                "actual_answer": actual_answer,
                "response_time": processing_time,
                "source_id": "",
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "evaluation_score": evaluation["score"],
                "evaluation_feedback": evaluation["feedback"]
            }])
            df = pd.concat([df, new_row], ignore_index=True)
            
            # Lưu kết quả vào file Excel sau mỗi 5 câu hỏi
            if i % 5 == 0 or i == len(benchmark_data):
                df.to_excel(output_file, index=False)
                print(f"\nĐã lưu kết quả của {i} câu hỏi vào file {output_file}")
            
        except Exception as e:
            print(f"\nLỗi khi xử lý câu hỏi {i}: {query}")
            print(f"Lỗi: {str(e)}")
            # Lưu kết quả lỗi
            result = BenchmarkResult(
                query=query,
                expected_answer=expected_answer,
                actual_answer="Lỗi khi xử lý câu hỏi",
                evaluation_score=0,
                evaluation_feedback=f"Lỗi: {str(e)}",
                relevant_docs=[],
                processing_time=0
            )
            results.append(result)
            
            # Thêm kết quả lỗi vào DataFrame
            new_row = pd.DataFrame([{
                "query": query,
                "expected_answer": expected_answer,
                "actual_answer": "Lỗi khi xử lý câu hỏi",
                "response_time": 0,
                "source_id": "",
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "evaluation_score": 0,
                "evaluation_feedback": f"Lỗi: {str(e)}"
            }])
            df = pd.concat([df, new_row], ignore_index=True)
            
            # Lưu kết quả lỗi vào file Excel
            df.to_excel(output_file, index=False)
            
    return results

def print_benchmark_results(results: List[BenchmarkResult]):
    """In kết quả benchmark"""
    print("\n=== KẾT QUẢ BENCHMARK ===")
    
    # Tính toán các chỉ số tổng hợp
    total_score = sum(r.evaluation_score for r in results)
    avg_score = total_score / len(results)
    total_time = sum(r.processing_time for r in results)
    avg_time = total_time / len(results)
    
    print(f"\nTổng số câu hỏi: {len(results)}")
    print(f"Điểm trung bình: {avg_score:.2f}/100")
    print(f"Thời gian trung bình: {avg_time:.2f} giây")
    
    # In chi tiết từng câu hỏi
    print("\n=== CHI TIẾT TỪNG CÂU HỎI ===")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Câu hỏi: {result.query}")
        print(f"   Câu trả lời mong đợi: {result.expected_answer}")
        print(f"   Câu trả lời thực tế: {result.actual_answer}")
        print(f"   Điểm đánh giá: {result.evaluation_score}/100")
        print(f"   Thời gian xử lý: {result.processing_time:.2f} giây")
        print(f"   Số tài liệu liên quan: {len(result.relevant_docs)}")
        print(f"   Phản hồi đánh giá: {result.evaluation_feedback}")

def main():
    # Load environment variables
    load_dotenv()
    
    # Get configuration from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    top_k = int(os.getenv("TOP_K", "3"))
    vector_weight = float(os.getenv("VECTOR_WEIGHT", "0.6"))
    
    # Verify API key is set
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please create a .env file with your API key.")
    
    # Set API key
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Data Ingestion Layer
    print("\n=== ĐANG XỬ LÝ DỮ LIỆU ===")
    documents = load_faq_data("data/TinhNangApp.json")
    processed_documents = preprocess_faq_data(documents)
    
    # Load benchmark data
    print("\n=== ĐANG ĐỌC DỮ LIỆU BENCHMARK ===")
    benchmark_data = load_benchmark_data("data/benchmark_TinhNangApp.json")
    
    # Embedding Layer
    print("\n=== ĐANG TẠO VECTOR DATABASE ===")
    vectordb = create_vectordb(processed_documents)
    
    # Retrieval Layer
    print("\n=== ĐANG CẤU HÌNH RETRIEVER ===")
    retriever = DocumentRetriever(
        vector_store=vectordb,
        documents=processed_documents,
        retriever_type="hybrid",
        hybrid_weights=[vector_weight, 1 - vector_weight],
        k=top_k
    )
    
    # Generation Layer
    print("\n=== ĐANG CẤU HÌNH GENERATOR ===")
    generator = AnswerGenerator(
        model_name=model_name,
        temperature=0.0,
        system_prompt=SYSTEM_PROMPT
    )
    
    # Evaluation Layer
    print("\n=== ĐANG CẤU HÌNH EVALUATOR ===")
    evaluator = RAGEvaluator(
        model_name=model_name,
        temperature=0.0,
        evaluation_prompt=EVALUATION_PROMPT
    )
    
    # Run benchmark
    print("\n=== ĐANG CHẠY BENCHMARK ===")
    results = run_benchmark(retriever, generator, evaluator, benchmark_data)
    
    # Print results
    print_benchmark_results(results)

if __name__ == "__main__":
    main()