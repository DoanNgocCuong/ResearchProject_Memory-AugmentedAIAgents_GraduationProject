import json
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
from pydantic import Field, BaseModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time

# Cài đặt thư viện cần thiết
# pip install langchain langchain-openai langchain-community chromadb rank_bm25 jq

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document, BaseRetriever
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import JSONLoader
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# Remove or replace the API key with environment variable
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

@dataclass
class SearchResult:
    """Lưu trữ kết quả tìm kiếm với score và document"""
    document: Document
    score: float

# 1. Tạo hàm đọc dữ liệu FAQ từ JSON
def load_faq_data(file_path: str) -> List[Document]:
    """Đọc dữ liệu FAQ từ file JSON"""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    documents = []
    for item in data:
        # Create a Document for each FAQ entry
        doc = Document(
            page_content=item['content'],
            metadata={
                "id": item.get("id"),
                "category": item.get("meta_data", {}).get("category", ""),
                "topic": item.get("meta_data", {}).get("topic", "")
            }
        )
        documents.append(doc)
    
    return documents

# 2. Tách câu hỏi và câu trả lời từ nội dung FAQ
def extract_qa_from_faq(content: str) -> Tuple[str, str]:
    """Tách câu hỏi và câu trả lời từ nội dung FAQ"""
    match = re.match(r"Q:\s*(.*?)\s*A:\s*(.*)", content, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "", ""

# 3. Tiền xử lý dữ liệu FAQ thành định dạng phù hợp hơn
def preprocess_faq_data(documents: List[Document]) -> List[Document]:
    """Tiền xử lý dữ liệu FAQ thành định dạng phù hợp hơn"""
    processed_docs = []
    
    for doc in documents:
        question, answer = extract_qa_from_faq(doc.page_content)
        
        # Tạo document cho câu hỏi với metadata phù hợp
        question_doc = Document(
            page_content=question,
            metadata={
                "id": doc.metadata.get("id"),
                "type": "question",
                "category": doc.metadata.get("category", ""),
                "topic": doc.metadata.get("topic", ""),
                "original_content": doc.page_content
            }
        )
        
        # Tạo document cho câu trả lời với metadata phù hợp
        answer_doc = Document(
            page_content=answer,
            metadata={
                "id": doc.metadata.get("id"),
                "type": "answer",
                "category": doc.metadata.get("category", ""),
                "topic": doc.metadata.get("topic", ""),
                "original_content": doc.page_content
            }
        )
        
        # Thêm vào danh sách
        processed_docs.extend([question_doc, answer_doc])
    
    return processed_docs

# 4. Tạo vector database từ documents
def create_vectordb(documents: List[Document]) -> Chroma:
    """Tạo vector database từ documents"""
    # Sử dụng OpenAI embeddings
    embeddings = OpenAIEmbeddings()
    
    # Tạo Chroma database với documents và embeddings
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db"  # Lưu database vào thư mục này
    )
    
    return vectordb

# 5. Tạo BM25 Retriever từ documents
def create_bm25_retriever(documents: List[Document], k: int = 3) -> BM25Retriever:
    """Tạo BM25 Retriever từ documents"""
    return BM25Retriever.from_documents(documents, k=k)

# 6. Tạo Hybrid Retriever kết hợp BM25 và Vector Search
def create_hybrid_retriever(vectordb: Chroma, bm25_retriever: BM25Retriever, 
                           vector_weight: float = 0.5, k: int = 3) -> EnsembleRetriever:
    """Tạo Hybrid Retriever kết hợp BM25 và Vector Search"""
    # Tạo vector retriever
    vector_retriever = vectordb.as_retriever(search_kwargs={"k": k})
    
    # Tạo ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[1-vector_weight, vector_weight]
    )
    
    return ensemble_retriever

# 7. Tạo QA chain với Hybrid Retriever
def create_hybrid_qa_chain(retriever: BaseRetriever) -> RetrievalQA:
    """Tạo QA chain với Hybrid Retriever"""
    # Sử dụng OpenAI language model
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    # Định nghĩa template cho prompt
    template = """
    Bạn là trợ lý hỗ trợ khách hàng của Robot Pika, một robot học tiếng Anh cho trẻ em.
    Hãy trả lời câu hỏi của khách hàng dựa trên thông tin sau đây:
    
    {context}
    
    Câu hỏi: {question}
    
    Trả lời một cách lịch sự, ngắn gọn và rõ ràng. Nếu không có thông tin trong context, hãy thông báo bạn không có thông tin để trả lời câu hỏi đó.
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # Tạo chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain

# 8. Tạo custom hybrid search function với re-ranking
def custom_hybrid_search(query: str, vectordb: Chroma, documents: List[Document], 
                       top_k: int = 5, vector_weight: float = 0.6) -> List[Document]:
    """
    Tìm kiếm sử dụng kết hợp vector search và semantic matching, sau đó re-rank kết quả
    
    Args:
        query: Câu truy vấn
        vectordb: Vector database
        documents: Danh sách documents gốc
        top_k: Số lượng kết quả trả về
        vector_weight: Trọng số cho vector search (từ 0 đến 1)
    
    Returns:
        List[Document]: Danh sách documents đã được sắp xếp theo độ liên quan
    """
    # 1. Vector search
    vector_results = vectordb.similarity_search_with_score(query, k=top_k*2)
    
    # Chuyển đổi thành SearchResult
    vector_search_results = [
        SearchResult(document=doc, score=score)
        for doc, score in vector_results
    ]
    
    # 2. Keyword search với BM25
    bm25_retriever = BM25Retriever.from_documents(documents, k=top_k*2)
    bm25_results = bm25_retriever.get_relevant_documents(query)
    
    # Tạo một dict để tra cứu nhanh document theo id và type
    doc_lookup = {}
    for doc in documents:
        key = (doc.metadata.get("id"), doc.metadata.get("type", ""))
        doc_lookup[key] = doc
    
    # Chuyển đổi thành SearchResult (giả định điểm từ 0.5 đến 1.0 cho BM25)
    bm25_search_results = []
    for i, doc in enumerate(bm25_results):
        normalized_score = 1.0 - (i / len(bm25_results)) * 0.5  # Từ 1.0 đến 0.5
        bm25_search_results.append(SearchResult(document=doc, score=normalized_score))
    
    # 3. Kết hợp và re-rank kết quả
    all_results = {}
    
    # Thêm kết quả từ vector search
    for result in vector_search_results:
        doc_id = result.document.metadata.get("id")
        doc_type = result.document.metadata.get("type", "")
        key = (doc_id, doc_type)
        
        if key not in all_results:
            all_results[key] = {
                "document": result.document,
                "vector_score": result.score,
                "bm25_score": 0.0
            }
        else:
            all_results[key]["vector_score"] = result.score
    
    # Thêm kết quả từ BM25 search
    for result in bm25_search_results:
        doc_id = result.document.metadata.get("id")
        doc_type = result.document.metadata.get("type", "")
        key = (doc_id, doc_type)
        
        if key not in all_results:
            all_results[key] = {
                "document": result.document,
                "vector_score": 0.0,
                "bm25_score": result.score
            }
        else:
            all_results[key]["bm25_score"] = result.score
    
    # Tính điểm tổng hợp
    for key, result in all_results.items():
        result["final_score"] = (
            result["vector_score"] * vector_weight +
            result["bm25_score"] * (1 - vector_weight)
        )
    
    # Sắp xếp theo điểm tổng hợp và lấy top_k kết quả
    sorted_results = sorted(
        all_results.values(),
        key=lambda x: x["final_score"],
        reverse=True
    )
    
    # Lọc ra các document duy nhất theo ID (ưu tiên document có điểm cao hơn)
    unique_doc_ids = set()
    final_results = []
    
    for result in sorted_results:
        doc_id = result["document"].metadata.get("id")
        if doc_id not in unique_doc_ids:
            unique_doc_ids.add(doc_id)
            
            # Lấy document gốc từ lookup
            original_doc_id = result["document"].metadata.get("id")
            original_content = result["document"].metadata.get("original_content")
            
            # Nếu có original_content, sử dụng nó
            if original_content:
                # Tạo document mới với nội dung gốc
                doc = Document(
                    page_content=original_content,
                    metadata=result["document"].metadata
                )
                final_results.append(doc)
            else:
                final_results.append(result["document"])
                
            # Dừng khi đủ top_k kết quả
            if len(final_results) >= top_k:
                break
    
    return final_results

# 9. Tạo custom retriever sử dụng hybrid search
class CustomHybridRetriever(BaseRetriever, BaseModel):
    """Custom retriever sử dụng hybrid search và re-ranking"""
    
    vectordb: Chroma = Field(description="Vector database for semantic search")
    documents: List[Document] = Field(description="List of documents for BM25 search")
    top_k: int = Field(default=3, description="Number of documents to retrieve")
    vector_weight: float = Field(default=0.6, description="Weight for vector search results")
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(self, query: str, *, run_manager: Optional[Any] = None) -> List[Document]:
        """Phương thức chính để lấy documents liên quan"""
        return custom_hybrid_search(
            query=query,
            vectordb=self.vectordb,
            documents=self.documents,
            top_k=self.top_k,
            vector_weight=self.vector_weight
        )
    
    async def _aget_relevant_documents(self, query: str, *, run_manager: Optional[Any] = None) -> List[Document]:
        """Async version of get_relevant_documents"""
        return self._get_relevant_documents(query, run_manager=run_manager)

def load_test_data(file_path: str) -> List[Dict[str, Any]]:
    """Load test dataset from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def evaluate_qa_system(qa_chain, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate QA system performance"""
    results = []
    correct_predictions = 0
    total_queries = len(test_data)
    total_time = 0
    
    # Store actual and predicted for metrics calculation
    actual_ids = []
    predicted_ids = []
    
    for i, test_case in enumerate(test_data, 1):
        query = test_case["query"]
        expected_id = test_case["expected_answer_id"]
        expected_answer = test_case["expected_answer"]
        
        # Time the response
        start_time = time.time()
        response = qa_chain.invoke({"query": query})
        end_time = time.time()
        
        # Get the response and source documents
        answer = response['result']
        source_docs = response.get('source_documents', [])
        
        # Get the ID of the first retrieved document
        if source_docs:
            predicted_id = source_docs[0].metadata.get('id')
        else:
            predicted_id = None
            
        # Calculate timing
        response_time = end_time - start_time
        total_time += response_time
        
        # Check if the prediction is correct
        is_correct = predicted_id == expected_id
        if is_correct:
            correct_predictions += 1
            
        # Store for metrics calculation
        actual_ids.append(expected_id)
        predicted_ids.append(predicted_id if predicted_id is not None else -1)
        
        # Store detailed results
        result = {
            "query": query,
            "expected_id": expected_id,
            "predicted_id": predicted_id,
            "expected_answer": expected_answer,
            "actual_answer": answer,
            "is_correct": is_correct,
            "response_time": response_time
        }
        results.append(result)
        
        # Print progress
        print(f"\nProcessing query {i}/{total_queries}")
        print(f"Query: {query}")
        print(f"Expected Answer: {expected_answer}")
        print(f"Actual Answer: {answer}")
        print(f"Correct: {is_correct}")
        print(f"Response Time: {response_time:.2f}s")
        print("-" * 80)
    
    # Calculate metrics
    accuracy = correct_predictions / total_queries
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual_ids, 
        predicted_ids, 
        average='weighted',
        zero_division=0
    )
    avg_response_time = total_time / total_queries
    
    metrics = {
        "total_queries": total_queries,
        "correct_predictions": correct_predictions,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "average_response_time": avg_response_time,
        "detailed_results": results
    }
    
    return metrics

def print_evaluation_results(metrics: Dict[str, Any]):
    """Print evaluation results in a formatted way"""
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total Queries: {metrics['total_queries']}")
    print(f"Correct Predictions: {metrics['correct_predictions']}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall: {metrics['recall']:.2%}")
    print(f"F1 Score: {metrics['f1_score']:.2%}")
    print(f"Average Response Time: {metrics['average_response_time']:.2f}s")
    print("\nDetailed Results:")
    print("-"*50)
    
    for result in metrics['detailed_results']:
        print(f"\nQuery: {result['query']}")
        print(f"Expected ID: {result['expected_id']}")
        print(f"Predicted ID: {result['predicted_id']}")
        print(f"Correct: {result['is_correct']}")
        print(f"Response Time: {result['response_time']:.2f}s")
        print("-"*30)

# Update your main function to include testing
def main():
    # First, save your test data to a file
    test_data = [
  {
    "query": "Khi nào tôi sẽ nhận được Robot Pika nếu đặt mua pre-order?",
    "expected_answer_id": 0,
    "expected_answer": "Robot dự kiến hoàn thiện trong 4 tháng (+ tối đa 2 tháng chênh lệch). Nếu sau 4 tháng vẫn chưa giao, bạn có quyền yêu cầu hoàn tiền."
  },
  {
    "query": "Thời gian giao hàng tối đa của Pre-order là bao lâu?",
    "expected_answer_id": 0,
    "expected_answer": "Robot dự kiến hoàn thiện trong 4 tháng (+ tối đa 2 tháng chênh lệch). Nếu sau 4 tháng vẫn chưa giao, bạn có quyền yêu cầu hoàn tiền."
  },
  {
    "query": "Tôi cần đặt cọc bao nhiêu khi đăng ký Pre-order?",
    "expected_answer_id": 1,
    "expected_answer": "Tùy vào đợt Pre-order, bạn có thể cần đặt cọc một khoản cố định. Thông tin cụ thể sẽ có trong hướng dẫn thanh toán."
  },
  {
    "query": "Robot Pika phù hợp cho độ tuổi và trình độ tiếng Anh nào?",
    "expected_answer_id": 2,
    "expected_answer": "Phiên bản đầu tiên tối ưu cho trẻ ở trình độ A1–A1.5 (tương đương Movers – Flyers). Trình độ phù hợp nhất sẽ nằm trong khoảng từ 5 - 10 tuổi."
  },
  {
    "query": "Nhà tôi có hai bé, có cần mua hai robot riêng không?",
    "expected_answer_id": 3,
    "expected_answer": "Phiên bản hiện tại tối ưu cho 1 người dùng chính. Nếu cần cho 2 bé, có thể có phụ phí hoặc tuỳ chọn nâng cấp trong tương lai."
  },
  {
    "query": "Sau năm đầu, phí subscription của Robot là bao nhiêu?",
    "expected_answer_id": 4,
    "expected_answer": "Năm đầu tiên đã bao gồm trong giá bán (gói 2.250k). Từ năm thứ hai, phí sẽ là 599k/năm nếu muốn duy trì toàn bộ tính năng."
  },
  {
    "query": "Robot được bảo hành như thế nào?",
    "expected_answer_id": 5,
    "expected_answer": "Bảo hành 1 năm cho lỗi do nhà sản xuất. Các lỗi do người dùng sẽ tính phí sửa chữa."
  },
  {
    "query": "Việc phụ huynh hỗ trợ con sử dụng Robot có phức tạp không?",
    "expected_answer_id": 6,
    "expected_answer": "Robot giao tiếp với bé, phụ huynh chỉ cần cài app và cài đặt ban đầu. Có hỗ trợ thêm nếu cần."
  },
  {
    "query": "Tôi muốn huỷ đơn Pre-order có được hoàn cọc không?",
    "expected_answer_id": 7,
    "expected_answer": "Có thể huỷ nếu Robot chưa vào sản xuất, hoàn cọc theo chính sách công ty."
  },
  {
    "query": "Nội dung học của Pika có linh hoạt tuỳ sở thích của bé không?",
    "expected_answer_id": 8,
    "expected_answer": "Có thể tuỳ chỉnh nội dung theo sở thích bé."
  },
  {
    "query": "Sau khi nhận Robot, tôi phải cài đặt như thế nào?",
    "expected_answer_id": 9,
    "expected_answer": "Kết nối với Wi-Fi và app điện thoại, Robot sẽ tự động cập nhật."
  },
  {
    "query": "Robot sửa phát âm tiếng Anh của bé bằng cách nào?",
    "expected_answer_id": 10,
    "expected_answer": "Pika dùng nhận diện giọng nói để sửa lỗi phát âm và hướng dẫn luyện tập."
  },
  {
    "query": "Tôi có thể liên hệ hỗ trợ kênh nào nếu có thắc mắc về Robot?",
    "expected_answer_id": 11,
    "expected_answer": "Qua hotline, email hoặc Zalo để được hỗ trợ."
  },
  {
    "query": "Robot có dùng được khi offline không?",
    "expected_answer_id": 12,
    "expected_answer": "Một số chức năng cơ bản hoạt động offline, nhưng cần Wi-Fi để cập nhật nội dung."
  },
  {
    "query": "Nếu Robot báo lỗi phần mềm, tôi cần làm gì?",
    "expected_answer_id": 13,
    "expected_answer": "Robot hỗ trợ cập nhật OTA và kỹ thuật viên hỗ trợ khi cần."
  },
  {
    "query": "Pika có giúp bé phát triển kỹ năng nào ngoài tiếng Anh không?",
    "expected_answer_id": 14,
    "expected_answer": "Có hỗ trợ kỹ năng giao tiếp, EQ và tư duy sáng tạo."
  },
  {
    "query": "Nếu làm rơi Robot hỏng, có thể thay linh kiện ở đâu?",
    "expected_answer_id": 15,
    "expected_answer": "Có thể liên hệ trung tâm hỗ trợ để sửa chữa hoặc thay linh kiện."
  },
  {
    "query": "Có phụ kiện nào kèm theo khi mua Robot không?",
    "expected_answer_id": 16,
    "expected_answer": "Robot đã kèm phụ kiện cơ bản, phụ kiện nâng cấp có bán riêng."
  },
  {
    "query": "Tôi có thể giới hạn thời gian bé chơi Robot mỗi ngày không?",
    "expected_answer_id": 17,
    "expected_answer": "Phụ huynh có thể cài đặt giới hạn qua app."
  },
  {
    "query": "Robot có được cập nhật bài học mới thường xuyên không?",
    "expected_answer_id": 18,
    "expected_answer": "Có, nội dung được cập nhật mỗi tháng nếu còn thời gian subscription."
  },
  {
    "query": "Khi bé học xong cấp độ hiện tại, có gói nâng cấp cao hơn không?",
    "expected_answer_id": 19,
    "expected_answer": "Sẽ có các gói mở rộng A2, B1 trong tương lai."
  },
  {
    "query": "Robot có thể cá nhân hoá bài học cho từng bé không?",
    "expected_answer_id": 20,
    "expected_answer": "Có, nội dung được điều chỉnh theo sở thích và tốc độ học của bé."
  },
  {
    "query": "Robot hỗ trợ nhiều tài khoản người dùng không?",
    "expected_answer_id": 21,
    "expected_answer": "Hiện ưu tiên 1 tài khoản chính, nhưng app quản lý nhiều bé, sẽ nâng cấp thêm profile trong tương lai."
  },
  {
    "query": "Điều gì xảy ra nếu tôi không gia hạn subscription?",
    "expected_answer_id": 4,
    "expected_answer": "Năm đầu tiên đã bao gồm trong giá bán (gói 2.250k). Từ năm thứ hai, phí sẽ là 599k/năm nếu muốn duy trì toàn bộ tính năng."
  },
  {
    "query": "Robot sẽ hoạt động ra sao khi mất kết nối Wi-Fi?",
    "expected_answer_id": 12,
    "expected_answer": "Một số chức năng cơ bản hoạt động offline, nhưng cần Wi-Fi để cập nhật nội dung."
  },
  {
    "query": "Bảo hành có áp dụng nếu Robot bị nước vào do tôi bất cẩn không?",
    "expected_answer_id": 5,
    "expected_answer": "Bảo hành 1 năm cho lỗi do nhà sản xuất. Các lỗi do người dùng sẽ tính phí sửa chữa."
  },
  {
    "query": "Trong hộp Robot Pika gồm những phụ kiện gì?",
    "expected_answer_id": 16,
    "expected_answer": "Robot đã kèm phụ kiện cơ bản, phụ kiện nâng cấp có bán riêng."
  },
  {
    "query": "Tôi muốn đặt giới hạn 30 phút sử dụng mỗi ngày, làm thế nào?",
    "expected_answer_id": 17,
    "expected_answer": "Phụ huynh có thể cài đặt giới hạn qua app."
  },
  {
    "query": "Pika thúc đẩy EQ của trẻ qua hoạt động nào?",
    "expected_answer_id": 14,
    "expected_answer": "Có hỗ trợ kỹ năng giao tiếp, EQ và tư duy sáng tạo."
  },
  {
    "query": "Làm sao Robot điều chỉnh nội dung theo sở thích của bé?",
    "expected_answer_id": 20,
    "expected_answer": "Có, nội dung được điều chỉnh theo sở thích và tốc độ học của bé."
  }
]

    import json
    with open("test_data.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    # Load and process FAQ data
    documents = load_faq_data("pika_faq.json")
    processed_docs = preprocess_faq_data(documents)
    
    # Create vector database
    vectordb = create_vectordb(processed_docs)
    
    # Create custom hybrid retriever
    hybrid_retriever = CustomHybridRetriever(
        vectordb=vectordb,
        documents=processed_docs,
        top_k=3,
        vector_weight=0.6
    )
    
    # Create QA chain
    qa_chain = create_hybrid_qa_chain(hybrid_retriever)
    
    # Load test data
    test_data = load_test_data("test_data.json")
    
    # Run evaluation
    print("Starting evaluation...")
    metrics = evaluate_qa_system(qa_chain, test_data)
    
    # Print results
    print_evaluation_results(metrics)

if __name__ == "__main__":
    main()