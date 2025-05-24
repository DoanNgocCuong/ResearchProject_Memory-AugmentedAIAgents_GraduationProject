import os
import json
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
from pydantic import Field, BaseModel

from layers._01_data_ingestion.loader import load_faq_data, preprocess_faq_data
from layers._03_embedding.embedder import DocumentEmbedder
from layers._04_retrieval.retriever import DocumentRetriever
from layers._05_generation.generator import AnswerGenerator
from layers._06_evaluation.evaluator import RAGEvaluator
from test_data import TEST_DATA

# Cài đặt thư viện cần thiết
# pip install langchain langchain-openai langchain-community chromadb rank_bm25 jq

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
class SearchResult:
    """Lưu trữ kết quả tìm kiếm với score và document"""
    document: Document
    score: float

class FeatureInfo(BaseModel):
    """Lưu trữ thông tin về tính năng"""
    feature_tag: str
    methods: List[str]
    application_values: List[str]
    content: str

def analyze_features(documents: List[Document]) -> Dict[str, FeatureInfo]:
    """Phân tích và nhóm các tính năng theo feature_tag"""
    features = {}
    for doc in documents:
        meta = doc.metadata
        if "feature_tag" in meta:
            tag = meta["feature_tag"]
            if tag not in features:
                features[tag] = FeatureInfo(
                    feature_tag=tag,
                    methods=meta.get("methods", []),
                    application_values=meta.get("application_values", []),
                    content=doc.page_content
                )
            else:
                # Cập nhật thông tin cho feature đã tồn tại
                features[tag].content += "\n" + doc.page_content
                features[tag].methods = list(set(features[tag].methods + meta.get("methods", [])))
                features[tag].application_values = list(set(features[tag].application_values + meta.get("application_values", [])))
    return features

def print_feature_analysis(features: Dict[str, FeatureInfo]):
    """In ra phân tích các tính năng"""
    print("\n=== PHÂN TÍCH TÍNH NĂNG ===")
    for tag, info in features.items():
        print(f"\nTính năng: {tag}")
        print(f"Phương pháp: {', '.join(info.methods)}")
        print(f"Giá trị ứng dụng: {', '.join(info.application_values)}")
        print(f"Nội dung: {info.content[:100]}...")

def create_vectordb(documents: List[Document], persist_directory: str = "data/chroma") -> FAISS:
    """
    Create a vector database from documents using OpenAI embeddings
    
    Args:
        documents: List of documents to embed
        persist_directory: Directory to store the vector database
        
    Returns:
        FAISS vector database instance
    """
    # Initialize embeddings
    embeddings = OpenAIEmbeddings()
    
    # Create vector database
    vectordb = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )
    
    return vectordb

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
    
    # Phân tích tính năng
    features = analyze_features(processed_documents)
    print_feature_analysis(features)
    
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
    
    # Interactive mode
    print("\n=== CHẾ ĐỘ TƯƠNG TÁC ===")
    print("Nhập 'exit' để thoát")
    while True:
        query = input("\nCâu hỏi của bạn: ")
        if query.lower() == 'exit':
            break
            
        # Tìm kiếm tài liệu liên quan
        relevant_docs = retriever.retrieve_documents(query)
        print("\nTài liệu liên quan:")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"\n{i}. {doc.page_content[:100]}...")
            
        # Tạo câu trả lời
        answer = generator.generate_answer(query, relevant_docs)
        print("\nCâu trả lời:", answer)
        
        # Đánh giá câu trả lời
        evaluation = evaluator.evaluate_answer(query, answer, relevant_docs)
        print("\nĐánh giá câu trả lời:")
        print(f"Điểm: {evaluation['score']}/100")
        print(f"Phản hồi: {evaluation['feedback']}")

if __name__ == "__main__":
    main()