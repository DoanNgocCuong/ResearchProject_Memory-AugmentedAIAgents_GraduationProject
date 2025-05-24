"""
Module đánh giá hiệu quả của khâu generation trong hệ thống RAG.
Kiểm tra chất lượng câu trả lời được sinh ra từ tài liệu và câu hỏi.
Sử dụng các metrics: BLEU-1, BLEU-2, BLEU-3, BLEU-4, Rouge-L, F1, LLM-Score
"""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os
import re
import json
from collections import Counter
import numpy as np

# Load environment variables
load_dotenv()

# Default generation evaluation prompt
DEFAULT_GENERATION_PROMPT = """Bạn là chuyên gia đánh giá chất lượng câu trả lời của hệ thống RAG.

Đánh giá câu trả lời và trả về kết quả dưới dạng JSON với format CHÍNH XÁC sau:

{{
    "overall_score": [điểm từ 0-100],
    "detailed_scores": {{
        "accuracy": [điểm từ 0-100],
        "completeness": [điểm từ 0-100], 
        "relevance": [điểm từ 0-100],
        "clarity": [điểm từ 0-100],
        "consistency": [điểm từ 0-100]
    }},
    "feedback": {{
        "accuracy_reason": "Giải thích về độ chính xác",
        "completeness_reason": "Giải thích về độ đầy đủ",
        "relevance_reason": "Giải thích về độ liên quan", 
        "clarity_reason": "Giải thích về độ rõ ràng",
        "consistency_reason": "Giải thích về tính nhất quán",
        "overall_assessment": "Đánh giá tổng thể"
    }}
}}

Tiêu chí đánh giá:
- accuracy: Câu trả lời có đúng với thông tin trong tài liệu không?
- completeness: Câu trả lời có trả lời đầy đủ câu hỏi không?
- relevance: Câu trả lời có liên quan trực tiếp đến câu hỏi không?
- clarity: Câu trả lời có dễ hiểu và rõ ràng không?
- consistency: Câu trả lời có nhất quán với thông tin nguồn không?

CHỈ trả về JSON, không thêm text nào khác."""

class GenerationEvaluator:
    """
    Class đánh giá hiệu quả của khâu generation trong RAG system.
    
    Tính năng:
    - Đánh giá tính chính xác của câu trả lời
    - Tính toán BLEU scores (1-4 grams)
    - Tính toán Rouge-L score
    - Tính toán F1 score
    - LLM-based evaluation
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
        documents: List[Document],
        reference_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Đánh giá tổng thể chất lượng câu trả lời với đầy đủ metrics.
        
        Args:
            question: Câu hỏi gốc
            answer: Câu trả lời được sinh ra
            documents: Danh sách tài liệu tham khảo
            reference_answer: Câu trả lời chuẩn (optional, để tính BLEU/Rouge)
            
        Returns:
            Dict chứa kết quả đánh giá chi tiết với tất cả metrics
        """
        context = "\n\n".join(doc.page_content for doc in documents)
        
        # LLM-based evaluation với JSON response
        llm_evaluation_json = self.chain.invoke({
            "context": context,
            "question": question,
            "answer": answer
        })
        
        # Parse JSON response
        try:
            evaluation_data = self._parse_json_response(llm_evaluation_json)
            llm_score = evaluation_data.get("overall_score", 0)
            detailed_scores = evaluation_data.get("detailed_scores", {})
            feedback_json = evaluation_data.get("feedback", {})
        except Exception as e:
            print(f"⚠️ JSON parsing failed: {e}")
            print(f"Raw response: {llm_evaluation_json}")
            # Fallback to old method
            llm_score = self._extract_score_fallback(llm_evaluation_json)
            detailed_scores = self._extract_detailed_scores_fallback(llm_evaluation_json)
            feedback_json = {"error": "Failed to parse JSON", "raw_response": llm_evaluation_json}
        
        # Prepare result dictionary
        result = {
            "llm_score": llm_score,
            "detailed_scores": detailed_scores,
            "llm_feedback": llm_evaluation_json,
            "feedback_structured": feedback_json,
            "context_used": context,
            "missing_information": self._find_missing_information(question, answer, documents),
            "hallucinations": self._detect_hallucinations(answer, documents),
            "answer_quality": self._assess_answer_quality(answer)
        }
        
        # Calculate reference-based metrics if reference answer is provided
        if reference_answer:
            reference_metrics = self._calculate_reference_metrics(answer, reference_answer)
            result.update(reference_metrics)
        else:
            # Calculate context-based metrics using documents as reference
            context_metrics = self._calculate_context_metrics(answer, context)
            result.update(context_metrics)
        
        return result
    
    def _calculate_reference_metrics(
        self,
        generated_answer: str,
        reference_answer: str
    ) -> Dict[str, float]:
        """
        Tính toán các metrics dựa trên câu trả lời tham chiếu.
        
        Args:
            generated_answer: Câu trả lời được sinh ra
            reference_answer: Câu trả lời chuẩn
            
        Returns:
            Dict chứa BLEU-1, BLEU-2, BLEU-3, BLEU-4, Rouge-L, F1
        """
        # Tokenize
        gen_tokens = self._tokenize(generated_answer)
        ref_tokens = self._tokenize(reference_answer)
        
        metrics = {}
        
        # Calculate BLEU scores (1-4 grams)
        for n in range(1, 5):
            bleu_score = self._calculate_bleu(gen_tokens, ref_tokens, n)
            metrics[f"bleu_{n}"] = round(bleu_score, 4)
        
        # Calculate Rouge-L
        rouge_l = self._calculate_rouge_l(gen_tokens, ref_tokens)
        metrics["rouge_l"] = round(rouge_l, 4)
        
        # Calculate F1
        f1_score = self._calculate_f1(gen_tokens, ref_tokens)
        metrics["f1"] = round(f1_score, 4)
        
        return metrics
    
    def _calculate_context_metrics(
        self,
        generated_answer: str,
        context: str
    ) -> Dict[str, float]:
        """
        Tính toán metrics dựa trên context khi không có reference answer.
        
        Args:
            generated_answer: Câu trả lời được sinh ra
            context: Nội dung tài liệu tham khảo
            
        Returns:
            Dict chứa các metrics tương tự
        """
        # Sử dụng context làm reference thay thế
        return self._calculate_reference_metrics(generated_answer, context)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text thành các từ."""
        # Xử lý tiếng Việt và tiếng Anh
        text = text.lower()
        # Loại bỏ dấu câu và ký tự đặc biệt
        text = re.sub(r'[^\w\s]', ' ', text)
        # Tách từ
        tokens = text.split()
        return [token for token in tokens if token.strip()]
    
    def _get_ngrams(self, tokens: List[str], n: int) -> List[tuple]:
        """Tạo n-grams từ danh sách tokens."""
        if len(tokens) < n:
            return []
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def _calculate_bleu(
        self,
        generated_tokens: List[str],
        reference_tokens: List[str],
        n: int
    ) -> float:
        """
        Tính BLEU score cho n-gram cụ thể.
        
        Args:
            generated_tokens: Tokens của câu trả lời sinh ra
            reference_tokens: Tokens của câu trả lời tham chiếu
            n: Độ dài n-gram
            
        Returns:
            BLEU score từ 0 đến 1
        """
        if len(generated_tokens) == 0 or len(reference_tokens) == 0:
            return 0.0
        
        # Tạo n-grams
        gen_ngrams = self._get_ngrams(generated_tokens, n)
        ref_ngrams = self._get_ngrams(reference_tokens, n)
        
        if not gen_ngrams or not ref_ngrams:
            return 0.0
        
        # Đếm n-grams
        gen_counter = Counter(gen_ngrams)
        ref_counter = Counter(ref_ngrams)
        
        # Tính số n-grams match
        matches = 0
        for ngram, count in gen_counter.items():
            if ngram in ref_counter:
                matches += min(count, ref_counter[ngram])
        
        # Tính precision
        if len(gen_ngrams) == 0:
            return 0.0
        
        precision = matches / len(gen_ngrams)
        
        # Brevity penalty (đơn giản hóa)
        bp = 1.0
        if len(generated_tokens) < len(reference_tokens):
            bp = np.exp(1 - len(reference_tokens) / len(generated_tokens))
        
        return bp * precision
    
    def _calculate_rouge_l(
        self,
        generated_tokens: List[str],
        reference_tokens: List[str]
    ) -> float:
        """
        Tính Rouge-L score dựa trên Longest Common Subsequence (LCS).
        
        Args:
            generated_tokens: Tokens của câu trả lời sinh ra
            reference_tokens: Tokens của câu trả lời tham chiếu
            
        Returns:
            Rouge-L F1 score từ 0 đến 1
        """
        if len(generated_tokens) == 0 and len(reference_tokens) == 0:
            return 1.0
        if len(generated_tokens) == 0 or len(reference_tokens) == 0:
            return 0.0
        
        # Tính LCS length
        lcs_length = self._lcs_length(generated_tokens, reference_tokens)
        
        # Tính precision và recall
        precision = lcs_length / len(generated_tokens) if len(generated_tokens) > 0 else 0
        recall = lcs_length / len(reference_tokens) if len(reference_tokens) > 0 else 0
        
        # Tính F1
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """
        Tính độ dài của Longest Common Subsequence.
        
        Args:
            seq1: Sequence thứ nhất
            seq2: Sequence thứ hai
            
        Returns:
            Độ dài LCS
        """
        m, n = len(seq1), len(seq2)
        
        # Tạo bảng DP
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill bảng DP
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _calculate_f1(
        self,
        generated_tokens: List[str],
        reference_tokens: List[str]
    ) -> float:
        """
        Tính F1 score dựa trên token overlap.
        
        Args:
            generated_tokens: Tokens của câu trả lời sinh ra
            reference_tokens: Tokens của câu trả lời tham chiếu
            
        Returns:
            F1 score từ 0 đến 1
        """
        if len(generated_tokens) == 0 and len(reference_tokens) == 0:
            return 1.0
        if len(generated_tokens) == 0 or len(reference_tokens) == 0:
            return 0.0
        
        # Chuyển thành set để tính intersection
        gen_set = set(generated_tokens)
        ref_set = set(reference_tokens)
        
        # Tính intersection
        intersection = gen_set.intersection(ref_set)
        
        # Tính precision và recall
        precision = len(intersection) / len(gen_set) if len(gen_set) > 0 else 0
        recall = len(intersection) / len(ref_set) if len(ref_set) > 0 else 0
        
        # Tính F1
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
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
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response từ LLM."""
        # Clean response - remove markdown code blocks if present
        cleaned_response = response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        
        # Try to parse JSON
        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            # Try to fix common JSON issues
            fixed_response = self._fix_json_response(cleaned_response)
            return json.loads(fixed_response)
    
    def _fix_json_response(self, response: str) -> str:
        """Cố gắng sửa các lỗi JSON phổ biến."""
        import re
        
        # Remove any text before first {
        response = re.sub(r'^[^{]*', '', response)
        # Remove any text after last }
        response = re.sub(r'}[^}]*$', '}', response)
        
        # Fix single quotes to double quotes
        response = response.replace("'", '"')
        
        # Fix trailing commas
        response = re.sub(r',(\s*[}\]])', r'\1', response)
        
        return response
    
    def _extract_score_fallback(self, evaluation_text: str) -> int:
        """Fallback method nếu JSON parsing thất bại."""
        import re
        
        # Tìm các pattern điểm số
        patterns = [
            r'overall_score["\']?\s*:\s*(\d+)',
            r'điểm tổng.*?(\d+)',
            r'overall.*?(\d+)',
            r'tổng.*?(\d+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, evaluation_text.lower())
            if matches:
                return min(int(matches[0]), 100)
        
        return 0
    
    def _extract_detailed_scores_fallback(self, evaluation_text: str) -> Dict[str, int]:
        """Fallback method cho detailed scores."""
        import re
        
        scores = {
            "accuracy": 0,
            "completeness": 0,
            "relevance": 0,
            "clarity": 0,
            "consistency": 0
        }
        
        # Improved patterns
        patterns = {
            "accuracy": [r'accuracy["\']?\s*:\s*(\d+)', r'chính xác["\']?\s*:\s*(\d+)'],
            "completeness": [r'completeness["\']?\s*:\s*(\d+)', r'đầy đủ["\']?\s*:\s*(\d+)'],
            "relevance": [r'relevance["\']?\s*:\s*(\d+)', r'liên quan["\']?\s*:\s*(\d+)'],
            "clarity": [r'clarity["\']?\s*:\s*(\d+)', r'rõ ràng["\']?\s*:\s*(\d+)'],
            "consistency": [r'consistency["\']?\s*:\s*(\d+)', r'nhất quán["\']?\s*:\s*(\d+)']
        }
        
        for criterion, criterion_patterns in patterns.items():
            for pattern in criterion_patterns:
                matches = re.findall(pattern, evaluation_text.lower())
                if matches:
                    scores[criterion] = min(int(matches[0]), 100)
                    break
        
        return scores

if __name__ == "__main__":
    """Test GenerationEvaluator với đầy đủ metrics"""
    
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
    
    # Reference answer for comparison
    reference_answer = "RAG (Retrieval-Augmented Generation) là phương pháp kết hợp tìm kiếm tài liệu với sinh text để tạo câu trả lời chính xác."
    
    print("Testing GenerationEvaluator với đầy đủ metrics...")
    try:
        evaluator = GenerationEvaluator()
        
        # Test with reference answer
        result = evaluator.evaluate_answer("RAG là gì?", sample_answer, sample_docs, reference_answer)
        
        print(f"LLM Score: {result['llm_score']}")
        print(f"BLEU-1: {result['bleu_1']}")
        print(f"BLEU-2: {result['bleu_2']}")
        print(f"BLEU-3: {result['bleu_3']}")
        print(f"BLEU-4: {result['bleu_4']}")
        print(f"Rouge-L: {result['rouge_l']}")
        print(f"F1: {result['f1']}")

        print(f"Detailed scores: {result['detailed_scores']}")
        print(f"Điểm Detailed scores này do LLMs trả ra, mục đích điểm này để phân tích chi tiết hơn.")
        print(f"Missing info: {result['missing_information']}")
        print(f"Hallucinations: {result['hallucinations']}")
        
        # Test without reference answer
        print("\n--- Test without reference answer ---")
        result_no_ref = evaluator.evaluate_answer("RAG là gì?", sample_answer, sample_docs)
        print(f"Context-based BLEU-1: {result_no_ref['bleu_1']}")
        print(f"Context-based Rouge-L: {result_no_ref['rouge_l']}")
        
    except Exception as e:
        print(f"Test failed: {e}")
