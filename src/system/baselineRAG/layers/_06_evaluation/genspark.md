Tôi sẽ giúp bạn refactor code thành 3 file riêng biệt để tách biệt việc đánh giá retrieval, generation và tổng hợp. Đây là cách tổ chức:

## 1. File `evaluator_retrieval.py` - Đánh giá khâu tìm kiếm tài liệu

```python
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
```

## 2. File `evaluator_generation.py` - Đánh giá khâu sinh câu trả lời

```python
"""
Module đánh giá hiệu quả của khâu generation trong hệ thống RAG.
Kiểm tra chất lượng câu trả lời được sinh ra từ tài liệu và câu hỏi.
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

# Default generation evaluation prompt
DEFAULT_GENERATION_PROMPT = """Bạn là chuyên gia đánh giá chất lượng câu trả lời của hệ thống RAG.
Nhiệm vụ của bạn là đánh giá câu trả lời được sinh ra dựa trên các tiêu chí sau:

1. Tính chính xác (0-100): Câu trả lời có đúng với thông tin trong tài liệu không?
2. Tính đầy đủ (0-100): Câu trả lời có trả lời đầy đủ câu hỏi không?
3. Tính liên quan (0-100): Câu trả lời có liên quan trực tiếp đến câu hỏi không?
4. Tính rõ ràng (0-100): Câu trả lời có dễ hiểu và rõ ràng không?
5. Tính nhất quán (0-100): Câu trả lời có nhất quán với thông tin nguồn không?

Cho điểm tổng từ 0 đến 100 và giải thích chi tiết từng tiêu chí."""

class GenerationEvaluator:
    """
    Class đánh giá hiệu quả của khâu generation trong RAG system.
    
    Tính năng:
    - Đánh giá tính chính xác của câu trả lời
    - Kiểm tra tính đầy đủ và liên quan
    - Phát hiện thông tin sai lệch hoặc bịa đặt
    - Đánh giá chất lượng ngôn ngữ
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        evaluation_prompt: str = DEFAULT_GENERATION_PROMPT
    ):
        """
        Khởi tạo GenerationEvaluator.
        
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
            ("human", """Tài liệu tham khảo:
            {context}
            
            Câu hỏi: {question}
            
            Câu trả lời được sinh ra: {answer}
            
            Đánh giá generation:""")
        ])
        
        self.chain = (
            {
                "context": RunnablePassthrough(),
                "question": RunnablePassthrough(),
                "answer": RunnablePassthrough()
            }
            | self.eval_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def evaluate_answer(
        self,
        question: str,
        answer: str,
        documents: List[Document]
    ) -> Dict[str, Any]:
        """
        Đánh giá tổng thể chất lượng câu trả lời.
        
        Args:
            question: Câu hỏi gốc
            answer: Câu trả lời được sinh ra
            documents: Danh sách tài liệu tham khảo
            
        Returns:
            Dict chứa kết quả đánh giá chi tiết
        """
        context = "\n\n".join(doc.page_content for doc in documents)
        
        evaluation = self.chain.invoke({
            "context": context,
            "question": question,
            "answer": answer
        })
        
        # Extract overall score
        overall_score = self._extract_score(evaluation)
        
        # Get detailed scores
        detailed_scores = self._extract_detailed_scores(evaluation)
        
        return {
            "overall_score": overall_score,
            "detailed_scores": detailed_scores,
            "feedback": evaluation,
            "context_used": context,
            "missing_information": self._find_missing_information(question, answer, documents),
            "hallucinations": self._detect_hallucinations(answer, documents),
            "answer_quality": self._assess_answer_quality(answer)
        }
    
    def _find_missing_information(
        self,
        question: str,
        answer: str,
        documents: List[Document]
    ) -> List[str]:
        """Tìm thông tin quan trọng bị thiếu trong câu trả lời."""
        context = "\n\n".join(doc.page_content for doc in documents)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Tìm ra thông tin quan trọng có trong tài liệu nhưng bị thiếu trong câu trả lời.
            Chỉ liệt kê những thông tin thực sự quan trọng để trả lời đầy đủ câu hỏi.
            Mỗi điểm thiếu viết trên một dòng."""),
            ("human", """Tài liệu: {context}
            
            Câu hỏi: {question}
            
            Câu trả lời: {answer}
            
            Thông tin bị thiếu:""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        missing_info = chain.invoke({
            "context": context,
            "question": question,
            "answer": answer
        })
        
        return [line.strip() for line in missing_info.split("\n") if line.strip()]
    
    def _detect_hallucinations(
        self,
        answer: str,
        documents: List[Document]
    ) -> List[str]:
        """Phát hiện thông tin bịa đặt (hallucination) trong câu trả lời."""
        context = "\n\n".join(doc.page_content for doc in documents)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Tìm ra những thông tin trong câu trả lời KHÔNG có trong tài liệu tham khảo.
            Chỉ liệt kê những thông tin cụ thể, rõ ràng là không có trong nguồn.
            Mỗi điểm hallucination viết trên một dòng."""),
            ("human", """Tài liệu tham khảo: {context}
            
            Câu trả lời: {answer}
            
            Thông tin bịa đặt:""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        hallucinations = chain.invoke({
            "context": context,
            "answer": answer
        })
        
        return [line.strip() for line in hallucinations.split("\n") if line.strip()]
    
    def _assess_answer_quality(self, answer: str) -> Dict[str, Any]:
        """Đánh giá chất lượng ngôn ngữ của câu trả lời."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Đánh giá chất lượng ngôn ngữ của câu trả lời theo các tiêu chí:
            1. Độ dài phù hợp (quá ngắn/quá dài)
            2. Cấu trúc câu rõ ràng
            3. Ngữ pháp chính xác
            4. Từ vựng phù hợp
            5. Tính logic trong trình bày
            
            Cho điểm từng tiêu chí và tổng kết."""),
            ("human", """Câu trả lời: {answer}
            
            Đánh giá chất lượng ngôn ngữ:""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        quality_assessment = chain.invoke({"answer": answer})
        
        return {
            "length": len(answer.split()),
            "assessment": quality_assessment,
            "readability_score": self._calculate_readability(answer)
        }
    
    def _calculate_readability(self, text: str) -> int:
        """Tính điểm đọc hiểu đơn giản."""
        words = len(text.split())
        sentences = text.count('.') + text.count('!') + text.count('?')
        
        if sentences == 0:
            return 50  # Điểm trung bình nếu không có câu
        
        avg_words_per_sentence = words / sentences
        
        # Điểm readability đơn giản dựa trên độ dài câu
        if avg_words_per_sentence <= 15:
            return 90
        elif avg_words_per_sentence <= 20:
            return 80
        elif avg_words_per_sentence <= 25:
            return 70
        elif avg_words_per_sentence <= 30:
            return 60
        else:
            return 50
    
    def _extract_score(self, evaluation_text: str) -> int:
        """Trích xuất điểm tổng từ text đánh giá."""
        score = 0
        for line in evaluation_text.split("\n"):
            if "tổng" in line.lower() or "overall" in line.lower():
                try:
                    import re
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        score = min(int(numbers[0]), 100)
                        break
                except:
                    continue
        return score
    
    def _extract_detailed_scores(self, evaluation_text: str) -> Dict[str, int]:
        """Trích xuất điểm chi tiết cho từng tiêu chí."""
        scores = {
            "accuracy": 0,
            "completeness": 0,
            "relevance": 0,
            "clarity": 0,
            "consistency": 0
        }
        
        criteria_map = {
            "chính xác": "accuracy",
            "đầy đủ": "completeness", 
            "liên quan": "relevance",
            "rõ ràng": "clarity",
            "nhất quán": "consistency"
        }
        
        for line in evaluation_text.split("\n"):
            for keyword, criterion in criteria_map.items():
                if keyword in line.lower():
                    try:
                        import re
                        numbers = re.findall(r'\d+', line)
                        if numbers:
                            scores[criterion] = min(int(numbers[0]), 100)
                    except:
                        continue
        
        return scores

if __name__ == "__main__":
    """Test GenerationEvaluator"""
    
    # Sample data
    sample_docs = [
        Document(
            page_content="RAG là viết tắt của Retrieval-Augmented Generation. Đây là phương pháp kết hợp việc tìm kiếm tài liệu liên quan với khả năng sinh text của mô hình ngôn ngữ để tạo ra câu trả lời chính xác hơn.",
            metadata={"source": "wiki_rag"}
        ),
        Document(
            page_content="Hệ thống RAG hoạt động bằng cách đầu tiên tìm kiếm các tài liệu liên quan đến câu hỏi, sau đó sử dụng thông tin từ những tài liệu này để sinh ra câu trả lời.",
            metadata={"source": "ai_guide"}
        )
    ]
    
    sample_answer = "RAG là một hệ thống AI giúp tạo ra câu trả lời tốt hơn bằng cách sử dụng thông tin từ tài liệu."
    
    print("Testing GenerationEvaluator...")
    try:
        evaluator = GenerationEvaluator()
        
        result = evaluator.evaluate_answer("RAG là gì?", sample_answer, sample_docs)
        
        print(f"Điểm tổng: {result['overall_score']}")
        print(f"Điểm chi tiết: {result['detailed_scores']}")
        print(f"Thông tin thiếu: {result['missing_information']}")
        print(f"Hallucinations: {result['hallucinations']}")
        print(f"Chất lượng ngôn ngữ: {result['answer_quality']['readability_score']}")
        
    except Exception as e:
        print(f"Test failed: {e}")
```

## 3. File `evaluation.py` - Đánh giá tổng hợp

```python
"""
Module đánh giá tổng hợp hệ thống RAG.
Kết hợp đánh giá retrieval và generation để đưa ra nhận định tổng thể.
"""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from evaluator_retrieval import RetrievalEvaluator
from evaluator_generation import GenerationEvaluator
import json
from datetime import datetime

class RAGEvaluator:
    """
    Class đánh giá tổng hợp hệ thống RAG.
    
    Tính năng:
    - Đánh giá tổng thể cả retrieval và generation
    - Tạo báo cáo chi tiết
    - So sánh hiệu suất giữa các lần chạy
    - Đưa ra khuyến nghị cải thiện
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0
    ):
        """
        Khởi tạo RAGEvaluator.
        
        Args:
            model_name: Tên model AI sử dụng
            temperature: Độ sáng tạo (0.0 - 1.0)
        """
        self.retrieval_evaluator = RetrievalEvaluator(
            model_name=model_name,
            temperature=temperature
        )
        self.generation_evaluator = GenerationEvaluator(
            model_name=model_name,
            temperature=temperature
        )
        
    def evaluate_full_pipeline(
        self,
        question: str,
        answer: str,
        documents: List[Document],
        retrieval_weight: float = 0.4,
        generation_weight: float = 0.6
    ) -> Dict[str, Any]:
        """
        Đánh giá toàn bộ pipeline RAG.
        
        Args:
            question: Câu hỏi gốc
            answer: Câu trả lời được sinh ra
            documents: Danh sách tài liệu được retrieve
            retrieval_weight: Trọng số cho điểm retrieval
            generation_weight: Trọng số cho điểm generation
            
        Returns:
            Dict chứa kết quả đánh giá tổng hợp
        """
        # Đánh giá retrieval
        retrieval_result = self.retrieval_evaluator.evaluate_retrieval(question, documents)
        
        # Đánh giá generation
        generation_result = self.generation_evaluator.evaluate_answer(question, answer, documents)
        
        # Tính điểm tổng hợp
        overall_score = (
            retrieval_result["relevance_score"] * retrieval_weight +
            generation_result["overall_score"] * generation_weight
        )
        
        # Tạo báo cáo tổng hợp
        evaluation_result = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "overall_score": round(overall_score, 2),
            "weights": {
                "retrieval": retrieval_weight,
                "generation": generation_weight
            },
            "retrieval_evaluation": retrieval_result,
            "generation_evaluation": generation_result,
            "summary": self._generate_summary(retrieval_result, generation_result, overall_score),
            "recommendations": self._generate_recommendations(retrieval_result, generation_result)
        }
        
        return evaluation_result
    
    def batch_evaluate(
        self,
        test_cases: List[Dict[str, Any]],
        save_results: bool = True,
        output_file: str = "rag_evaluation_results.json"
    ) -> Dict[str, Any]:
        """
        Đánh giá hàng loạt nhiều test case.
        
        Args:
            test_cases: List các dict chứa 'question', 'answer', 'documents'
            save_results: Có lưu kết quả ra file không
            output_file: Tên file để lưu kết quả
            
        Returns:
            Dict chứa kết quả tổng hợp tất cả test case
        """
        results = []
        total_scores = {
            "overall": [],
            "retrieval": [],
            "generation": []
        }
        
        for i, test_case in enumerate(test_cases):
            print(f"Đang đánh giá test case {i+1}/{len(test_cases)}...")
            
            result = self.evaluate_full_pipeline(
                question=test_case["question"],
                answer=test_case["answer"],
                documents=test_case["documents"]
            )
            
            results.append(result)
            total_scores["overall"].append(result["overall_score"])
            total_scores["retrieval"].append(result["retrieval_evaluation"]["relevance_score"])
            total_scores["generation"].append(result["generation_evaluation"]["overall_score"])
        
        # Tính thống kê tổng hợp
        batch_summary = {
            "total_test_cases": len(test_cases),
            "timestamp": datetime.now().isoformat(),
            "average_scores": {
                "overall": round(sum(total_scores["overall"]) / len(total_scores["overall"]), 2),
                "retrieval": round(sum(total_scores["retrieval"]) / len(total_scores["retrieval"]), 2),
                "generation": round(sum(total_scores["generation"]) / len(total_scores["generation"]), 2)
            },
            "score_distribution": self._calculate_score_distribution(total_scores),
            "individual_results": results
        }
        
        if save_results:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(batch_summary, f, ensure_ascii=False, indent=2)
            print(f"Kết quả đã được lưu vào {output_file}")
        
        return batch_summary
    
    def _generate_summary(
        self,
        retrieval_result: Dict[str, Any],
        generation_result: Dict[str, Any],
        overall_score: float
    ) -> Dict[str, str]:
        """Tạo tóm tắt đ