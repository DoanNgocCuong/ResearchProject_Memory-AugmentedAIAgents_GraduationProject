"""
Module đánh giá tổng hợp hệ thống RAG.
Kết hợp đánh giá retrieval và generation để đưa ra nhận định tổng thể.
Hỗ trợ đầy đủ các metrics: BLEU-1,2,3,4, Rouge-L, F1, LLM-Score
"""

from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from evaluator_retrieval import RetrievalEvaluator
from evaluator_generation import GenerationEvaluator
import json
import os
import csv
from datetime import datetime
from dataclasses import dataclass
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

@dataclass
class TestCase:
    """Data class cho một test case đánh giá RAG."""
    question: str
    answer: str
    documents: List[Document]
    reference_answer: Optional[str] = None
    category: Optional[str] = None
    difficulty: Optional[str] = "medium"  # easy, medium, hard
    expected_score: Optional[float] = None

class RAGEvaluator:
    """
    Class đánh giá tổng hợp hệ thống RAG.
    
    Tính năng:
    - Đánh giá tổng thể cả retrieval và generation
    - Tạo báo cáo chi tiết với đầy đủ metrics
    - So sánh hiệu suất giữa các lần chạy
    - Đưa ra khuyến nghị cải thiện
    - Xuất báo cáo dạng JSON, CSV, HTML
    - Tạo visualization charts
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        output_dir: str = "evaluation_results"
    ):
        """
        Khởi tạo RAGEvaluator.
        
        Args:
            model_name: Tên model AI sử dụng
            temperature: Độ sáng tạo (0.0 - 1.0)
            output_dir: Thư mục lưu kết quả đánh giá
        """
        self.retrieval_evaluator = RetrievalEvaluator(
            model_name=model_name,
            temperature=temperature
        )
        self.generation_evaluator = GenerationEvaluator(
            model_name=model_name,
            temperature=temperature
        )
        
        self.output_dir = output_dir
        self._ensure_output_dir()
        
    def _ensure_output_dir(self):
        """Tạo thư mục output nếu chưa tồn tại."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Đã tạo thư mục: {self.output_dir}")
        
    def evaluate_full_pipeline(
        self,
        question: str,
        answer: str,
        documents: List[Document],
        reference_answer: Optional[str] = None,
        retrieval_weight: float = 0.4,
        generation_weight: float = 0.6,
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Đánh giá toàn bộ pipeline RAG với đầy đủ metrics.
        
        Args:
            question: Câu hỏi gốc
            answer: Câu trả lời được sinh ra
            documents: Danh sách tài liệu được retrieve
            reference_answer: Câu trả lời chuẩn (optional)
            retrieval_weight: Trọng số cho điểm retrieval
            generation_weight: Trọng số cho điểm generation
            category: Phân loại câu hỏi (optional)
            
        Returns:
            Dict chứa kết quả đánh giá tổng hợp với đầy đủ metrics
        """
        print(f"Đang đánh giá câu hỏi: {question[:50]}...")
        
        # Đánh giá retrieval
        print("- Đánh giá retrieval...")
        retrieval_result = self.retrieval_evaluator.evaluate_retrieval(question, documents)
        
        # Đánh giá generation
        print("- Đánh giá generation...")
        generation_result = self.generation_evaluator.evaluate_answer(
            question, answer, documents, reference_answer
        )
        
        # Tính điểm tổng hợp
        overall_score = (
            retrieval_result["relevance_score"] * retrieval_weight +
            generation_result["llm_score"] * generation_weight
        )
        
        # Tạo báo cáo tổng hợp
        evaluation_result = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "answer": answer,
                "reference_answer": reference_answer,
                "category": category,
                "document_count": len(documents)
            },
            "scores": {
                "overall_score": round(overall_score, 2),
                "retrieval_score": retrieval_result["relevance_score"],
                "generation_llm_score": generation_result["llm_score"]
            },
            "weights": {
                "retrieval": retrieval_weight,
                "generation": generation_weight
            },
            "metrics": self._extract_metrics(generation_result),
            "detailed_scores": generation_result.get("detailed_scores", {}),
            "retrieval_evaluation": retrieval_result,
            "generation_evaluation": generation_result,
            "analysis": {
                "summary": self._generate_summary(retrieval_result, generation_result, overall_score),
                "recommendations": self._generate_recommendations(retrieval_result, generation_result),
                "strengths": self._identify_strengths(retrieval_result, generation_result),
                "weaknesses": self._identify_weaknesses(retrieval_result, generation_result)
            }
        }
        
        print(f"- Hoàn thành. Điểm tổng: {overall_score:.1f}/100")
        return evaluation_result
    
    def _extract_metrics(self, generation_result: Dict[str, Any]) -> Dict[str, float]:
        """Trích xuất các metrics từ kết quả generation."""
        metrics = {}
        
        # BLEU scores
        for i in range(1, 5):
            key = f"bleu_{i}"
            if key in generation_result:
                metrics[key] = generation_result[key]
        
        # Other metrics
        for metric in ["rouge_l", "f1"]:
            if metric in generation_result:
                metrics[metric] = generation_result[metric]
        
        return metrics
    
    def batch_evaluate(
        self,
        test_cases: List[TestCase],
        save_results: bool = True,
        output_prefix: str = "rag_evaluation",
        create_visualizations: bool = True
    ) -> Dict[str, Any]:
        """
        Đánh giá hàng loạt nhiều test case với báo cáo chi tiết.
        
        Args:
            test_cases: List các TestCase
            save_results: Có lưu kết quả ra file không
            output_prefix: Prefix cho tên file output
            create_visualizations: Có tạo biểu đồ không
            
        Returns:
            Dict chứa kết quả tổng hợp tất cả test case
        """
        print(f"\n🚀 Bắt đầu đánh giá batch với {len(test_cases)} test cases...")
        
        results = []
        all_scores = {
            "overall": [],
            "retrieval": [],
            "generation": [],
            "bleu_1": [],
            "bleu_2": [],
            "bleu_3": [],
            "bleu_4": [],
            "rouge_l": [],
            "f1": []
        }
        
        categories_stats = {}
        
        for i, test_case in enumerate(test_cases):
            print(f"\n📝 Test case {i+1}/{len(test_cases)}")
            
            try:
                result = self.evaluate_full_pipeline(
                    question=test_case.question,
                    answer=test_case.answer,
                    documents=test_case.documents,
                    reference_answer=test_case.reference_answer,
                    category=test_case.category
                )
                
                results.append(result)
                
                # Collect scores
                all_scores["overall"].append(result["scores"]["overall_score"])
                all_scores["retrieval"].append(result["scores"]["retrieval_score"])
                all_scores["generation"].append(result["scores"]["generation_llm_score"])
                
                # Collect metrics
                metrics = result.get("metrics", {})
                for metric_name in ["bleu_1", "bleu_2", "bleu_3", "bleu_4", "rouge_l", "f1"]:
                    if metric_name in metrics:
                        all_scores[metric_name].append(metrics[metric_name])
                
                # Category statistics
                category = test_case.category or "unknown"
                if category not in categories_stats:
                    categories_stats[category] = {
                        "count": 0,
                        "scores": []
                    }
                categories_stats[category]["count"] += 1
                categories_stats[category]["scores"].append(result["scores"]["overall_score"])
                
            except Exception as e:
                print(f"❌ Lỗi khi đánh giá test case {i+1}: {e}")
                continue
        
        # Tính thống kê tổng hợp
        batch_summary = self._calculate_batch_statistics(
            results, all_scores, categories_stats, test_cases
        )
        
        if save_results:
            self._save_batch_results(batch_summary, output_prefix)
        
        if create_visualizations and results:
            self._create_visualizations(batch_summary, output_prefix)
        
        print(f"\n✅ Hoàn thành đánh giá batch!")
        print(f"📊 Điểm trung bình tổng thể: {batch_summary['statistics']['average_scores']['overall']:.1f}/100")
        
        return batch_summary
    
    def _calculate_batch_statistics(
        self,
        results: List[Dict],
        all_scores: Dict[str, List],
        categories_stats: Dict,
        test_cases: List[TestCase]
    ) -> Dict[str, Any]:
        """Tính toán thống kê tổng hợp cho batch evaluation."""
        
        def safe_mean(scores):
            return round(statistics.mean(scores), 2) if scores else 0
        
        def safe_median(scores):
            return round(statistics.median(scores), 2) if scores else 0
        
        def safe_stdev(scores):
            return round(statistics.stdev(scores), 2) if len(scores) > 1 else 0
        
        # Calculate category averages
        category_averages = {}
        for category, stats in categories_stats.items():
            if stats["scores"]:
                category_averages[category] = {
                    "average_score": safe_mean(stats["scores"]),
                    "count": stats["count"],
                    "min_score": min(stats["scores"]),
                    "max_score": max(stats["scores"])
                }
        
        batch_summary = {
            "metadata": {
                "total_test_cases": len(test_cases),
                "successful_evaluations": len(results),
                "failed_evaluations": len(test_cases) - len(results),
                "timestamp": datetime.now().isoformat(),
                "evaluation_duration": "N/A"  # Could be calculated if needed
            },
            "statistics": {
                "average_scores": {
                    "overall": safe_mean(all_scores["overall"]),
                    "retrieval": safe_mean(all_scores["retrieval"]),
                    "generation": safe_mean(all_scores["generation"]),
                    "bleu_1": safe_mean(all_scores["bleu_1"]),
                    "bleu_2": safe_mean(all_scores["bleu_2"]),
                    "bleu_3": safe_mean(all_scores["bleu_3"]),
                    "bleu_4": safe_mean(all_scores["bleu_4"]),
                    "rouge_l": safe_mean(all_scores["rouge_l"]),
                    "f1": safe_mean(all_scores["f1"])
                },
                "median_scores": {
                    "overall": safe_median(all_scores["overall"]),
                    "retrieval": safe_median(all_scores["retrieval"]),
                    "generation": safe_median(all_scores["generation"])
                },
                "score_distribution": self._calculate_score_distribution(all_scores),
                "standard_deviation": {
                    "overall": safe_stdev(all_scores["overall"]),
                    "retrieval": safe_stdev(all_scores["retrieval"]),
                    "generation": safe_stdev(all_scores["generation"])
                }
            },
            "category_analysis": category_averages,
            "performance_insights": self._generate_performance_insights(all_scores, category_averages),
            "individual_results": results
        }
        
        return batch_summary
    
    def _calculate_score_distribution(self, all_scores: Dict[str, List]) -> Dict[str, Dict]:
        """Tính phân bố điểm số."""
        distribution = {}
        
        for score_type, scores in all_scores.items():
            if not scores:
                continue
                
            # Define score ranges
            ranges = {
                "excellent": [90, 100],
                "good": [70, 89],
                "average": [50, 69],
                "poor": [0, 49]
            }
            
            dist = {range_name: 0 for range_name in ranges.keys()}
            
            for score in scores:
                for range_name, (min_val, max_val) in ranges.items():
                    if min_val <= score <= max_val:
                        dist[range_name] += 1
                        break
            
            # Convert to percentages
            total = len(scores)
            dist_percent = {k: round((v/total)*100, 1) for k, v in dist.items()}
            
            distribution[score_type] = {
                "counts": dist,
                "percentages": dist_percent
            }
        
        return distribution
    
    def _generate_performance_insights(
        self,
        all_scores: Dict[str, List],
        category_averages: Dict
    ) -> Dict[str, Any]:
        """Tạo insights về hiệu suất hệ thống."""
        insights = {
            "overall_performance": "",
            "best_category": "",
            "worst_category": "",
            "metric_analysis": {},
            "recommendations": []
        }
        
        # Overall performance assessment
        overall_avg = statistics.mean(all_scores["overall"]) if all_scores["overall"] else 0
        if overall_avg >= 80:
            insights["overall_performance"] = "Xuất sắc - Hệ thống hoạt động rất tốt"
        elif overall_avg >= 70:
            insights["overall_performance"] = "Tốt - Hệ thống hoạt động ổn định"
        elif overall_avg >= 60:
            insights["overall_performance"] = "Trung bình - Cần cải thiện một số điểm"
        else:
            insights["overall_performance"] = "Kém - Cần cải thiện đáng kể"
        
        # Best and worst categories
        if category_averages:
            best_cat = max(category_averages.items(), key=lambda x: x[1]["average_score"])
            worst_cat = min(category_averages.items(), key=lambda x: x[1]["average_score"])
            
            insights["best_category"] = f"{best_cat[0]} ({best_cat[1]['average_score']:.1f} điểm)"
            insights["worst_category"] = f"{worst_cat[0]} ({worst_cat[1]['average_score']:.1f} điểm)"
        
        # Metric analysis
        for metric in ["bleu_1", "bleu_2", "bleu_3", "bleu_4", "rouge_l", "f1"]:
            if all_scores[metric]:
                avg_score = statistics.mean(all_scores[metric])
                if avg_score < 0.3:
                    insights["metric_analysis"][metric] = "Thấp - Cần cải thiện"
                elif avg_score < 0.6:
                    insights["metric_analysis"][metric] = "Trung bình"
                else:
                    insights["metric_analysis"][metric] = "Tốt"
        
        # Generate recommendations
        if overall_avg < 70:
            insights["recommendations"].append("Tối ưu hóa cả retrieval và generation")
        
        ret_avg = statistics.mean(all_scores["retrieval"]) if all_scores["retrieval"] else 0
        gen_avg = statistics.mean(all_scores["generation"]) if all_scores["generation"] else 0
        
        if ret_avg < gen_avg - 10:
            insights["recommendations"].append("Tập trung cải thiện khâu retrieval")
        elif gen_avg < ret_avg - 10:
            insights["recommendations"].append("Tập trung cải thiện khâu generation")
        
        if all_scores["bleu_1"] and statistics.mean(all_scores["bleu_1"]) < 0.3:
            insights["recommendations"].append("Cải thiện độ chính xác từ vựng (BLEU scores thấp)")
        
        if all_scores["rouge_l"] and statistics.mean(all_scores["rouge_l"]) < 0.3:
            insights["recommendations"].append("Cải thiện cấu trúc câu trả lời (Rouge-L thấp)")
        
        return insights
    
    def _save_batch_results(self, batch_summary: Dict, output_prefix: str):
        """Lưu kết quả batch evaluation ra nhiều định dạng."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON
        json_file = os.path.join(self.output_dir, f"{output_prefix}_{timestamp}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(batch_summary, f, ensure_ascii=False, indent=2)
        print(f"📄 Đã lưu JSON: {json_file}")
        
        # Save CSV summary
        csv_file = os.path.join(self.output_dir, f"{output_prefix}_summary_{timestamp}.csv")
        self._save_csv_summary(batch_summary, csv_file)
        print(f"📊 Đã lưu CSV: {csv_file}")
        
        # Save detailed CSV
        detailed_csv = os.path.join(self.output_dir, f"{output_prefix}_detailed_{timestamp}.csv")
        self._save_detailed_csv(batch_summary, detailed_csv)
        print(f"📋 Đã lưu CSV chi tiết: {detailed_csv}")
    
    def _save_csv_summary(self, batch_summary: Dict, csv_file: str):
        """Lưu tóm tắt dạng CSV."""
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['Metric', 'Average', 'Median', 'Std Dev'])
            
            # Scores
            stats = batch_summary['statistics']
            for metric, avg in stats['average_scores'].items():
                median = stats['median_scores'].get(metric, 'N/A')
                std = stats['standard_deviation'].get(metric, 'N/A')
                writer.writerow([metric, avg, median, std])
    
    def _save_detailed_csv(self, batch_summary: Dict, csv_file: str):
        """Lưu kết quả chi tiết dạng CSV."""
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            header = [
                'Question', 'Category', 'Overall_Score', 'Retrieval_Score', 
                'Generation_Score', 'BLEU_1', 'BLEU_2', 'BLEU_3', 'BLEU_4',
                'Rouge_L', 'F1', 'Missing_Info_Count', 'Hallucination_Count'
            ]
            writer.writerow(header)
            
            # Data rows
            for result in batch_summary['individual_results']:
                row = [
                    result['metadata']['question'][:100] + '...' if len(result['metadata']['question']) > 100 else result['metadata']['question'],
                    result['metadata'].get('category', 'unknown'),
                    result['scores']['overall_score'],
                    result['scores']['retrieval_score'],
                    result['scores']['generation_llm_score'],
                    result['metrics'].get('bleu_1', 'N/A'),
                    result['metrics'].get('bleu_2', 'N/A'),
                    result['metrics'].get('bleu_3', 'N/A'),
                    result['metrics'].get('bleu_4', 'N/A'),
                    result['metrics'].get('rouge_l', 'N/A'),
                    result['metrics'].get('f1', 'N/A'),
                    len(result['generation_evaluation'].get('missing_information', [])),
                    len(result['generation_evaluation'].get('hallucinations', []))
                ]
                writer.writerow(row)
    
    def _create_visualizations(self, batch_summary: Dict, output_prefix: str):
        """Tạo các biểu đồ visualization."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # 1. Score distribution histogram
            self._plot_score_distribution(batch_summary, output_prefix, timestamp)
            
            # 2. Metrics comparison
            self._plot_metrics_comparison(batch_summary, output_prefix, timestamp)
            
            # 3. Category performance
            if batch_summary.get('category_analysis'):
                self._plot_category_performance(batch_summary, output_prefix, timestamp)
            
            print(f"📈 Đã tạo biểu đồ visualization")
            
        except Exception as e:
            print(f"⚠️ Không thể tạo visualization: {e}")
    
    def _plot_score_distribution(self, batch_summary: Dict, output_prefix: str, timestamp: str):
        """Vẽ biểu đồ phân bố điểm số."""
        scores_data = []
        labels = []
        
        for result in batch_summary['individual_results']:
            scores_data.append([
                result['scores']['overall_score'],
                result['scores']['retrieval_score'],
                result['scores']['generation_llm_score']
            ])
            labels.append('Overall,Retrieval,Generation'.split(','))
        
        if not scores_data:
            return
        
        # Flatten data for plotting
        all_scores = [score for scores in scores_data for score in scores]
        all_labels = [label for labels_row in labels for label in labels_row]
        
        plt.figure(figsize=(12, 6))
        
        # Subplot 1: Histogram
        plt.subplot(1, 2, 1)
        plt.hist(all_scores, bins=20, alpha=0.7, edgecolor='black')
        plt.title('Phân Bố Điểm Số Tổng Thể')
        plt.xlabel('Điểm Số')
        plt.ylabel('Tần Suất')
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Box plot
        plt.subplot(1, 2, 2)
        overall_scores = [result['scores']['overall_score'] for result in batch_summary['individual_results']]
        retrieval_scores = [result['scores']['retrieval_score'] for result in batch_summary['individual_results']]
        generation_scores = [result['scores']['generation_llm_score'] for result in batch_summary['individual_results']]
        
        plt.boxplot([overall_scores, retrieval_scores, generation_scores], 
                   labels=['Overall', 'Retrieval', 'Generation'])
        plt.title('So Sánh Điểm Số Theo Thành Phần')
        plt.ylabel('Điểm Số')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = os.path.join(self.output_dir, f"{output_prefix}_score_distribution_{timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metrics_comparison(self, batch_summary: Dict, output_prefix: str, timestamp: str):
        """Vẽ biểu đồ so sánh các metrics."""
        metrics_avg = batch_summary['statistics']['average_scores']
        
        # Filter out metrics with values
        plot_metrics = {k: v for k, v in metrics_avg.items() 
                       if k in ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 'rouge_l', 'f1'] and v > 0}
        
        if not plot_metrics:
            return
        
        plt.figure(figsize=(10, 6))
        
        metrics_names = list(plot_metrics.keys())
        metrics_values = list(plot_metrics.values())
        
        bars = plt.bar(metrics_names, metrics_values, alpha=0.8)
        plt.title('So Sánh Các Metrics Đánh Giá')
        plt.ylabel('Điểm Số')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plot_file = os.path.join(self.output_dir, f"{output_prefix}_metrics_comparison_{timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_category_performance(self, batch_summary: Dict, output_prefix: str, timestamp: str):
        """Vẽ biểu đồ hiệu suất theo category."""
        categories = batch_summary['category_analysis']
        
        if len(categories) < 2:
            return
        
        plt.figure(figsize=(10, 6))
        
        cat_names = list(categories.keys())
        cat_scores = [categories[cat]['average_score'] for cat in cat_names]
        cat_counts = [categories[cat]['count'] for cat in cat_names]
        
        # Create subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Average scores by category
        bars1 = ax1.bar(cat_names, cat_scores, alpha=0.8)
        ax1.set_title('Điểm Trung Bình Theo Category')
        ax1.set_ylabel('Điểm Số')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars1, cat_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{score:.1f}', ha='center', va='bottom')
        
        # Count by category
        bars2 = ax2.bar(cat_names, cat_counts, alpha=0.8, color='orange')
        ax2.set_title('Số Lượng Test Cases Theo Category')
        ax2.set_ylabel('Số Lượng')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, count in zip(bars2, cat_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        plot_file = os.path.join(self.output_dir, f"{output_prefix}_category_performance_{timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_summary(
        self,
        retrieval_result: Dict[str, Any],
        generation_result: Dict[str, Any],
        overall_score: float
    ) -> Dict[str, str]:
        """Tạo tóm tắt đánh giá với đầy đủ metrics."""
        
        # Extract metrics từ generation result
        metrics_summary = ""
        if 'bleu_1' in generation_result:
            metrics_summary = f"""
        Metrics chi tiết:
        - BLEU-1: {generation_result['bleu_1']:.3f}
        - BLEU-2: {generation_result['bleu_2']:.3f}
        - BLEU-3: {generation_result['bleu_3']:.3f}
        - BLEU-4: {generation_result['bleu_4']:.3f}
        - Rouge-L: {generation_result['rouge_l']:.3f}
        - F1: {generation_result['f1']:.3f}
        """
        
        summary = {
            "overall_assessment": f"Hệ thống RAG đạt điểm tổng thể: {overall_score:.1f}/100",
            "retrieval_summary": f"Retrieval: {retrieval_result['relevance_score']}/100 - " + 
                               ("Tốt" if retrieval_result['relevance_score'] >= 70 else 
                                "Trung bình" if retrieval_result['relevance_score'] >= 50 else "Cần cải thiện"),
            "generation_summary": f"Generation: {generation_result['llm_score']}/100 - " + 
                                ("Tốt" if generation_result['llm_score'] >= 70