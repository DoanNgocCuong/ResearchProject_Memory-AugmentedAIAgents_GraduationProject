"""
Module đánh giá hiệu quả của khâu retrieval trong hệ thống RAG.
Kiểm tra xem các tài liệu được tìm kiếm có phù hợp với câu hỏi không.
"""

from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Default retrieval evaluation prompt
DEFAULT_RETRIEVAL_PROMPT = """Bạn là chuyên gia đánh giá hệ thống retrieval.
Nhiệm vụ của bạn là kiểm tra xem các tài liệu được tìm kiếm có phù hợp với câu hỏi không.

Hãy đánh giá theo các tiêu chí sau:
1. Mức độ liên quan của tài liệu với câu hỏi (0-100)
2. Tính đầy đủ của thông tin
3. Chất lượng nguồn tài liệu
4. Tìm ra tài liệu không liên quan (nếu có)
5. Đề xuất tài liệu bị thiếu (nếu có)

Cho điểm từ 0 đến 100 và giải thích chi tiết."""

class RetrievalEvaluator:
    """
    Class đánh giá hiệu quả của khâu retrieval trong RAG system.
    
    Tính năng:
    - Đánh giá mức độ liên quan của tài liệu
    - Tìm tài liệu không phù hợp
    - Phát hiện thông tin bị thiếu
    - Đánh giá chất lượng nguồn tài liệu
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        evaluation_prompt: str = DEFAULT_RETRIEVAL_PROMPT
    ):
        """
        Khởi tạo RetrievalEvaluator.
        
        Args:
            model_name: Tên model AI sử dụng
            temperature: Độ sáng tạo (0.0 - 1.0)
            evaluation_prompt: Prompt hướng dẫn đánh giá
        """
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature
        )
        
        self.eval_prompt = ChatPromptTemplate.from_messages([
            ("system", evaluation_prompt),
            ("human", """Câu hỏi: {question}
            
            Các tài liệu được tìm kiếm:
            {documents}
            
            Đánh giá retrieval:""")
        ])
        
        self.chain = (
            {
                "question": RunnablePassthrough(),
                "documents": RunnablePassthrough()
            }
            | self.eval_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def evaluate_retrieval(
        self,
        question: str,
        documents: List[Document]
    ) -> Dict[str, Any]:
        """
        Đánh giá tổng thể khâu retrieval.
        
        Args:
            question: Câu hỏi gốc
            documents: Danh sách tài liệu được tìm kiếm
            
        Returns:
            Dict chứa kết quả đánh giá
        """
        if not documents:
            return {
                "relevance_score": 0,
                "feedback": "Không có tài liệu nào được tìm kiếm.",
                "retrieved_docs_count": 0,
                "irrelevant_docs": [],
                "missing_topics": []
            }
        
        # Format documents for evaluation
        doc_text = ""
        for i, doc in enumerate(documents, 1):
            doc_text += f"Tài liệu {i}:\n{doc.page_content}\n"
            if doc.metadata:
                doc_text += f"Metadata: {doc.metadata}\n"
            doc_text += "\n"
        
        evaluation = self.chain.invoke({
            "question": question,
            "documents": doc_text
        })
        
        # Extract score
        score = self._extract_score(evaluation)
        
        return {
            "relevance_score": score,
            "feedback": evaluation,
            "retrieved_docs_count": len(documents),
            "irrelevant_docs": self._find_irrelevant_docs(question, documents),
            "missing_topics": self._find_missing_topics(question, documents)
        }
    
    def evaluate_document_relevance(
        self,
        question: str,
        documents: List[Document]
    ) -> List[Dict[str, Any]]:
        """
        Đánh giá từng tài liệu riêng lẻ.
        
        Args:
            question: Câu hỏi gốc
            documents: Danh sách tài liệu
            
        Returns:
            List chứa đánh giá cho từng tài liệu
        """
        results = []
        
        for i, doc in enumerate(documents):
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Đánh giá mức độ liên quan của tài liệu này với câu hỏi. Cho điểm từ 0-100."),
                ("human", """Câu hỏi: {question}
                
                Tài liệu: {document}
                
                Đánh giá:""")
            ])
            
            chain = prompt | self.llm | StrOutputParser()
            
            evaluation = chain.invoke({
                "question": question,
                "document": doc.page_content
            })
            
            score = self._extract_score(evaluation)
            
            results.append({
                "document_index": i,
                "relevance_score": score,
                "feedback": evaluation,
                "metadata": doc.metadata
            })
        
        return results
    
    def _find_irrelevant_docs(
        self,
        question: str,
        documents: List[Document]
    ) -> List[int]:
        """Tìm các tài liệu không liên quan."""
        doc_evaluations = self.evaluate_document_relevance(question, documents)
        irrelevant = []
        
        for eval_result in doc_evaluations:
            if eval_result["relevance_score"] < 30:  # Threshold for irrelevant
                irrelevant.append(eval_result["document_index"])
        
        return irrelevant
    
    def _find_missing_topics(
        self,
        question: str,
        documents: List[Document]
    ) -> List[str]:
        """Tìm các chủ đề/thông tin bị thiếu."""
        context = "\n\n".join(doc.page_content for doc in documents)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Phân tích câu hỏi và tài liệu hiện có.
            Tìm ra những chủ đề/thông tin quan trọng còn thiếu để trả lời đầy đủ câu hỏi.
            Liệt kê từng điểm thiếu trên một dòng."""),
            ("human", """Câu hỏi: {question}
            
            Tài liệu hiện có: {context}
            
            Thông tin bị thiếu:""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        missing_info = chain.invoke({
            "question": question,
            "context": context
        })
        
        return [line.strip() for line in missing_info.split("\n") if line.strip()]
    
    def _extract_score(self, evaluation_text: str) -> int:
        """Trích xuất điểm số từ text đánh giá."""
        score = 0
        for line in evaluation_text.split("\n"):
            if "điểm" in line.lower() or "score" in line.lower():
                try:
                    # Tìm số trong dòng
                    import re
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        score = min(int(numbers[0]), 100)  # Cap at 100
                        break
                except:
                    continue
        return score

if __name__ == "__main__":
    """Test RetrievalEvaluator"""
    
    # Sample documents
    sample_docs = [
        Document(
            page_content="RAG là viết tắt của Retrieval-Augmented Generation. Đây là phương pháp kết hợp việc tìm kiếm tài liệu liên quan với khả năng sinh text của mô hình ngôn ngữ.",
            metadata={"source": "wiki_rag"}
        ),
        Document(
            page_content="Hệ thống RAG giúp mô hình AI đưa ra câu trả lời chính xác và cập nhật bằng cách sử dụng kiến thức từ nguồn dữ liệu bên ngoài.",
            metadata={"source": "ai_guide"}
        ),
        Document(
            page_content="Thời tiết hôm nay rất đẹp, nắng vàng và nhiệt độ khoảng 25 độ C.",
            metadata={"source": "weather_report"}
        )
    ]
    
    print("Testing RetrievalEvaluator...")
    try:
        evaluator = RetrievalEvaluator()
        
        # Test overall retrieval evaluation
        result = evaluator.evaluate_retrieval("RAG là gì?", sample_docs)
        print(f"Điểm retrieval: {result['relevance_score']}")
        print(f"Số tài liệu: {result['retrieved_docs_count']}")
        print(f"Tài liệu không liên quan: {result['irrelevant_docs']}")
        print(f"Thông tin thiếu: {result['missing_topics']}")
        
        # Test individual document evaluation
        print("\nĐánh giá từng tài liệu:")
        doc_results = evaluator.evaluate_document_relevance("RAG là gì?", sample_docs)
        for result in doc_results:
            print(f"Tài liệu {result['document_index']}: {result['relevance_score']} điểm")
            
    except Exception as e:
        print(f"Test failed: {e}")
