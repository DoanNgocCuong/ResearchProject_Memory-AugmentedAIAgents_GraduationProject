"""
File test cho hệ thống đánh giá RAG.
Chạy file này để test toàn bộ hệ thống với dữ liệu mẫu.
"""

import os
import sys
from typing import List
from langchain_core.documents import Document

# Import các module đánh giá
from evaluator_retrieval import RetrievalEvaluator
from evaluator_generation import GenerationEvaluator
from evaluation import RAGEvaluator, TestCase

def create_sample_documents() -> List[Document]:
    """Tạo tài liệu mẫu để test."""
    return [
        Document(
            page_content="""
            RAG (Retrieval-Augmented Generation) là một phương pháp trong AI kết hợp hai kỹ thuật:
            1. Retrieval: Tìm kiếm thông tin liên quan từ cơ sở dữ liệu
            2. Generation: Sinh ra câu trả lời dựa trên thông tin đã tìm được
            
            RAG được sử dụng rộng rãi trong các chatbot và hệ thống Q&A để cải thiện độ chính xác
            và cung cấp thông tin cập nhật mà không cần train lại model.
            """,
            metadata={"source": "ai_handbook", "page": 1, "topic": "RAG_basics"}
        ),
        Document(
            page_content="""
            Quy trình hoạt động của RAG:
            1. Người dùng đặt câu hỏi
            2. Hệ thống tìm kiếm tài liệu liên quan trong vector database
            3. Kết hợp câu hỏi và tài liệu tìm được
            4. Đưa vào Language Model để sinh câu trả lời
            5. Trả về câu trả lời cho người dùng
            
            Ưu điểm: Cập nhật thông tin real-time, giảm hallucination
            Nhược điểm: Phụ thuộc vào chất lượng retrieval
            """,
            metadata={"source": "ai_handbook", "page": 2, "topic": "RAG_process"}
        ),
        Document(
            page_content="""
            Các ứng dụng phổ biến của RAG:
            - Customer support chatbots
            - Internal knowledge management systems
            - Research assistants
            - Code documentation systems
            - Legal document analysis
            
            RAG đặc biệt hiệu quả cho các domain cần thông tin cập nhật thường xuyên
            như tin tức, chính sách, hướng dẫn kỹ thuật.
            """,
            metadata={"source": "ai_applications", "page": 5, "topic": "RAG_applications"}
        ),
        Document(
            page_content="""
            Thời tiết hôm nay ở Hà Nội khá đẹp với nắng vàng và nhiệt độ khoảng 25°C.
            Dự báo tuần tới sẽ có mưa rải rác vào chiều tối.
            Người dân nên chuẩn bị áo mưa khi ra đường.
            """,
            metadata={"source": "weather_report", "date": "2024-01-15"}
        )
    ]

def create_test_cases() -> List[TestCase]:
    """Tạo các test case để đánh giá."""
    documents = create_sample_documents()
    
    test_cases = [
        TestCase(
            question="RAG là gì?",
            answer="RAG (Retrieval-Augmented Generation) là phương pháp AI kết hợp tìm kiếm thông tin và sinh text để tạo câu trả lời chính xác hơn.",
            documents=documents,
            reference_answer="RAG là phương pháp kết hợp retrieval và generation để tạo ra câu trả lời chính xác dựa trên thông tin được tìm kiếm.",
            category="basic_concept",
            difficulty="easy"
        ),
        TestCase(
            question="RAG hoạt động như thế nào?",
            answer="RAG hoạt động qua 5 bước: nhận câu hỏi, tìm kiếm tài liệu, kết hợp thông tin, sinh câu trả lời qua LM, và trả kết quả cho user.",
            documents=documents,
            reference_answer="RAG hoạt động bằng cách tìm kiếm tài liệu liên quan, sau đó kết hợp với câu hỏi để đưa vào language model sinh ra câu trả lời.",
            category="process",
            difficulty="medium"
        ),
        TestCase(
            question="Ưu điểm và nhược điểm của RAG?",
            answer="Ưu điểm: cập nhật real-time, giảm hallucination. Nhược điểm: phụ thuộc chất lượng retrieval.",
            documents=documents,
            reference_answer="RAG có ưu điểm là cung cấp thông tin cập nhật và giảm thiểu hallucination, nhưng nhược điểm là phụ thuộc vào chất lượng của khâu retrieval.",
            category="analysis",
            difficulty="medium"
        ),
        TestCase(
            question="RAG được ứng dụng trong lĩnh vực nào?",
            answer="RAG được ứng dụng trong chatbot hỗ trợ khách hàng, hệ thống quản lý kiến thức, trợ lý nghiên cứu, và phân tích tài liệu pháp lý.",
            documents=documents,
            reference_answer="RAG được ứng dụng rộng rãi trong customer support, knowledge management, research assistance và legal document analysis.",
            category="applications",
            difficulty="easy"
        ),
        TestCase(
            question="Thời tiết hôm nay như thế nào?",
            answer="Hôm nay thời tiết đẹp với nắng vàng và nhiệt độ 25°C. Tuần tới có mưa rải rác.",
            documents=documents,
            reference_answer="Thời tiết hôm nay ở Hà Nội khá đẹp với nắng vàng và nhiệt độ khoảng 25°C.",
            category="irrelevant",
            difficulty="easy"
        )
    ]
    
    return test_cases

def test_retrieval_evaluator():
    """Test RetrievalEvaluator."""
    print("\n" + "="*60)
    print("🔍 TESTING RETRIEVAL EVALUATOR")
    print("="*60)
    
    try:
        evaluator = RetrievalEvaluator()
        documents = create_sample_documents()
        
        # Test 1: Câu hỏi liên quan
        print("\n📝 Test 1: Câu hỏi về RAG (liên quan)")
        result = evaluator.evaluate_retrieval("RAG là gì?", documents)
        
        print(f"✅ Điểm relevance: {result['relevance_score']}/100")
        print(f"📊 Số tài liệu: {result['retrieved_docs_count']}")
        print(f"🚫 Tài liệu không liên quan: {result['irrelevant_docs']}")
        print(f"❓ Thông tin thiếu: {len(result['missing_topics'])} điểm")
        
        # Test 2: Câu hỏi không liên quan  
        print("\n📝 Test 2: Câu hỏi về thời tiết (không liên quan)")
        result2 = evaluator.evaluate_retrieval("Thời tiết hôm nay như thế nào?", documents)
        
        print(f"✅ Điểm relevance: {result2['relevance_score']}/100")
        print(f"🚫 Tài liệu không liên quan: {result2['irrelevant_docs']}")
        
        # Test 3: Đánh giá từng tài liệu
        print("\n📝 Test 3: Đánh giá từng tài liệu riêng lẻ")
        doc_results = evaluator.evaluate_document_relevance("RAG là gì?", documents[:2])
        for i, doc_result in enumerate(doc_results):
            print(f"   Tài liệu {i}: {doc_result['relevance_score']}/100 điểm")
        
        print("\n✅ RetrievalEvaluator test completed!")
        
    except Exception as e:
        print(f"❌ RetrievalEvaluator test failed: {e}")

def test_generation_evaluator():
    """Test GenerationEvaluator."""
    print("\n" + "="*60)
    print("⚡ TESTING GENERATION EVALUATOR")
    print("="*60)
    
    try:
        evaluator = GenerationEvaluator()
        documents = create_sample_documents()
        
        # Test with reference answer
        print("\n📝 Test 1: Đánh giá với reference answer")
        question = "RAG là gì?"
        answer = "RAG là phương pháp AI kết hợp tìm kiếm và sinh text."
        reference = "RAG là phương pháp kết hợp retrieval và generation."
        
        result = evaluator.evaluate_answer(question, answer, documents, reference)
        
        print(f"✅ LLM Score: {result['llm_score']}/100")
        print(f"📊 BLEU-1: {result.get('bleu_1', 'N/A'):.3f}")
        print(f"📊 BLEU-2: {result.get('bleu_2', 'N/A'):.3f}")
        print(f"📊 BLEU-3: {result.get('bleu_3', 'N/A'):.3f}")
        print(f"📊 BLEU-4: {result.get('bleu_4', 'N/A'):.3f}")
        print(f"📊 Rouge-L: {result.get('rouge_l', 'N/A'):.3f}")
        print(f"📊 F1: {result.get('f1', 'N/A'):.3f}")
        print(f"❓ Missing info: {len(result['missing_information'])} điểm")
        print(f"🔥 Hallucinations: {len(result['hallucinations'])} điểm")
        
        # Test without reference answer
        print("\n📝 Test 2: Đánh giá không có reference answer")
        result2 = evaluator.evaluate_answer(question, answer, documents)
        
        print(f"✅ LLM Score: {result2['llm_score']}/100")
        print(f"📊 Context-based BLEU-1: {result2.get('bleu_1', 'N/A'):.3f}")
        print(f"📊 Context-based Rouge-L: {result2.get('rouge_l', 'N/A'):.3f}")
        
        print("\n✅ GenerationEvaluator test completed!")
        
    except Exception as e:
        print(f"❌ GenerationEvaluator test failed: {e}")

def test_full_rag_evaluator():
    """Test RAGEvaluator (đánh giá tổng hợp)."""
    print("\n" + "="*60)
    print("🚀 TESTING FULL RAG EVALUATOR")
    print("="*60)
    
    try:
        evaluator = RAGEvaluator(output_dir="test_results")
        documents = create_sample_documents()
        
        # Test single evaluation
        print("\n📝 Test 1: Đánh giá đơn lẻ")
        question = "RAG là gì?"
        answer = "RAG là phương pháp AI kết hợp tìm kiếm thông tin và sinh text."
        reference = "RAG là phương pháp kết hợp retrieval và generation."
        
        result = evaluator.evaluate_full_pipeline(
            question=question,
            answer=answer,
            documents=documents,
            reference_answer=reference,
            category="basic_concept"
        )
        
        print(f"✅ Điểm tổng thể: {result['scores']['overall_score']}/100")
        print(f"🔍 Điểm retrieval: {result['scores']['retrieval_score']}/100")
        print(f"⚡ Điểm generation: {result['scores']['generation_llm_score']}/100")
        
        metrics = result.get('metrics', {})
        if metrics:
            print(f"📊 BLEU-1: {metrics.get('bleu_1', 'N/A'):.3f}")
            print(f"📊 Rouge-L: {metrics.get('rouge_l', 'N/A'):.3f}")
            print(f"📊 F1: {metrics.get('f1', 'N/A'):.3f}")
        
        print(f"💡 Tóm tắt: {result['analysis']['summary']['overall_assessment']}")
        
        print("\n✅ Single evaluation test completed!")
        
    except Exception as e:
        print(f"❌ RAGEvaluator single test failed: {e}")

def test_batch_evaluation():
    """Test batch evaluation."""
    print("\n" + "="*60)
    print("📊 TESTING BATCH EVALUATION")
    print("="*60)
    
    try:
        evaluator = RAGEvaluator(output_dir="test_results")
        test_cases = create_test_cases()
        
        print(f"🎯 Chuẩn bị đánh giá {len(test_cases)} test cases...")
        
        # Run batch evaluation
        batch_result = evaluator.batch_evaluate(
            test_cases=test_cases,
            save_results=True,
            output_prefix="test_batch",
            create_visualizations=False  # Set False để tránh lỗi nếu thiếu matplotlib
        )
        
        print("\n📈 KẾT QUẢ BATCH EVALUATION:")
        print(f"✅ Tổng số test cases: {batch_result['metadata']['total_test_cases']}")
        print(f"✅ Đánh giá thành công: {batch_result['metadata']['successful_evaluations']}")
        print(f"❌ Đánh giá thất bại: {batch_result['metadata']['failed_evaluations']}")
        
        avg_scores = batch_result['statistics']['average_scores']
        print(f"\n📊 ĐIỂM TRUNG BÌNH:")
        print(f"   Tổng thể: {avg_scores['overall']:.1f}/100")
        print(f"   Retrieval: {avg_scores['retrieval']:.1f}/100")
        print(f"   Generation: {avg_scores['generation']:.1f}/100")
        print(f"   BLEU-1: {avg_scores.get('bleu_1', 0):.3f}")
        print(f"   Rouge-L: {avg_scores.get('rouge_l', 0):.3f}")
        print(f"   F1: {avg_scores.get('f1', 0):.3f}")
        
        # Category analysis
        if batch_result.get('category_analysis'):
            print(f"\n🏷️ PHÂN TÍCH THEO CATEGORY:")
            for category, stats in batch_result['category_analysis'].items():
                print(f"   {category}: {stats['average_score']:.1f}/100 ({stats['count']} cases)")
        
        # Performance insights
        insights = batch_result.get('performance_insights', {})
        if insights:
            print(f"\n💡 INSIGHTS:")
            print(f"   Đánh giá tổng thể: {insights.get('overall_performance', 'N/A')}")
            if insights.get('best_category'):
                print(f"   Category tốt nhất: {insights['best_category']}")
            if insights.get('worst_category'):
                print(f"   Category cần cải thiện: {insights['worst_category']}")
            
            if insights.get('recommendations'):
                print(f"\n📋 KHUYẾN NGHỊ:")
                for rec in insights['recommendations']:
                    print(f"   • {rec}")
        
        print("\n✅ Batch evaluation test completed!")
        print(f"📁 Kết quả đã được lưu trong thư mục: test_results/")
        
    except Exception as e:
        print(f"❌ Batch evaluation test failed: {e}")

def test_metrics_calculation():
    """Test tính toán các metrics riêng lẻ."""
    print("\n" + "="*60)
    print("🔢 TESTING METRICS CALCULATION")
    print("="*60)
    
    try:
        evaluator = GenerationEvaluator()
        
        # Test BLEU calculation
        print("\n📝 Test BLEU calculation:")
        gen_text = "RAG là phương pháp kết hợp tìm kiếm và sinh text"
        ref_text = "RAG là phương pháp kết hợp retrieval và generation"
        
        gen_tokens = evaluator._tokenize(gen_text)
        ref_tokens = evaluator._tokenize(ref_text)
        
        print(f"Generated tokens: {gen_tokens}")
        print(f"Reference tokens: {ref_tokens}")
        
        for n in range(1, 5):
            bleu = evaluator._calculate_bleu(gen_tokens, ref_tokens, n)
            print(f"BLEU-{n}: {bleu:.3f}")
        
        # Test Rouge-L calculation
        rouge_l = evaluator._calculate_rouge_l(gen_tokens, ref_tokens)
        print(f"Rouge-L: {rouge_l:.3f}")
        
        # Test F1 calculation
        f1 = evaluator._calculate_f1(gen_tokens, ref_tokens)
        print(f"F1: {f1:.3f}")
        
        print("\n✅ Metrics calculation test completed!")
        
    except Exception as e:
        print(f"❌ Metrics calculation test failed: {e}")

def main():
    """Chạy tất cả các test."""
    print("🎯 BẮT ĐẦU KIỂM TRA HỆ THỐNG ĐÁNH GIÁ RAG")
    print("="*80)
    
    # Tạo thư mục kết quả nếu chưa có
    os.makedirs("test_results", exist_ok=True)
    
    # Chạy từng test
    test_retrieval_evaluator()
    test_generation_evaluator()
    test_metrics_calculation()
    test_full_rag_evaluator()
    test_batch_evaluation()
    
    print("\n" + "="*80)
    print("🎉 HOÀN THÀNH TẤT CẢ KIỂM TRA!")
    print("="*80)
    print("📁 Kết quả test đã được lưu trong thư mục: test_results/")
    print("💡 Bạn có thể xem các file JSON, CSV để phân tích chi tiết.")

if __name__ == "__main__":
    # Kiểm tra dependencies
    required_modules = ['langchain_core', 'langchain_openai', 'openai', 'dotenv']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print("❌ Thiếu các module sau:")
        for module in missing_modules:
            print(f"   - {module}")
        print("\n💡 Cài đặt bằng lệnh: pip install langchain langchain-openai python-dotenv")
        sys.exit(1)
    
    # Kiểm tra API key
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️ Cảnh báo: Không tìm thấy OPENAI_API_KEY trong biến môi trường.")
        print("💡 Tạo file .env và thêm: OPENAI_API_KEY=your_api_key_here")
        
        # Có thể vẫn chạy test nhưng sẽ fail ở phần gọi API
        choice = input("Bạn có muốn tiếp tục test (có thể bị lỗi API)? (y/n): ")
        if choice.lower() != 'y':
            sys.exit(1)
    
    main()
