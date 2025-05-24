# RAG System Evaluator Documentation

## 1. Cài đặt và Khởi tạo

### 1.1. Cài đặt thư viện
```python
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

# Load biến môi trường
load_dotenv()
```

### 1.2. Khởi tạo Evaluator
```python
# Khởi tạo với cấu hình mặc định
evaluator = RAGEvaluator()

# Hoặc khởi tạo với cấu hình tùy chỉnh
evaluator = RAGEvaluator(
    model_name="gpt-4",        # Tên model OpenAI
    temperature=0.0,           # Độ sáng tạo (0.0 - 1.0)
    max_tokens=4096,          # Số token tối đa
    evaluation_prompt="..."    # Prompt tùy chỉnh
)
```

## 2. Chuẩn bị Dữ liệu

### 2.1. Tạo tài liệu
```python
# Tạo danh sách tài liệu
tai_lieu = [
    Document(
        page_content="RAG (Retrieval-Augmented Generation) là một kỹ thuật kết hợp việc truy xuất tài liệu với việc tạo câu trả lời bằng mô hình ngôn ngữ.",
        metadata={"source": "tai_lieu_1"}
    ),
    Document(
        page_content="RAG giúp mô hình ngôn ngữ đưa ra câu trả lời chính xác và cập nhật hơn bằng cách sử dụng kiến thức bên ngoài.",
        metadata={"source": "tai_lieu_2"}
    )
]
```

### 2.2. Chuẩn bị câu hỏi và câu trả lời
```python
cau_hoi = "RAG là gì?"
cau_tra_loi = "RAG là một hệ thống giúp mô hình AI đưa ra câu trả lời tốt hơn bằng cách sử dụng thông tin từ tài liệu."
```

## 3. Sử dụng Evaluator

### 3.1. Đánh giá câu trả lời
```python
# Đánh giá chất lượng câu trả lời
ket_qua = evaluator.evaluate_answer(
    question=cau_hoi,
    answer=cau_tra_loi,
    documents=tai_lieu
)

# In kết quả
print(f"Điểm số: {ket_qua['score']}")
print(f"Phản hồi: {ket_qua['feedback']}")
```

### 3.2. Tìm thông tin thiếu
```python
# Tìm thông tin bị thiếu trong câu trả lời
thong_tin_thieu = evaluator.find_missing_information(
    question=cau_hoi,
    answer=cau_tra_loi,
    documents=tai_lieu
)

# In thông tin thiếu
print("Thông tin bị thiếu:")
for item in thong_tin_thieu:
    print(f"- {item}")
```

### 3.3. Đánh giá việc truy xuất tài liệu
```python
# Đánh giá chất lượng truy xuất tài liệu
ket_qua_truy_xuat = evaluator.evaluate_retrieval(
    question=cau_hoi,
    documents=tai_lieu
)

# In kết quả
print(f"Điểm số liên quan: {ket_qua_truy_xuat['relevance_score']}")
print(f"Phản hồi: {ket_qua_truy_xuat['feedback']}")
print(f"Số tài liệu đã truy xuất: {ket_qua_truy_xuat['retrieved_docs']}")
```

## 4. Xử lý Lỗi

### 4.1. Try-Catch cơ bản
```python
try:
    ket_qua = evaluator.evaluate_answer(cau_hoi, cau_tra_loi, tai_lieu)
except Exception as e:
    print(f"Lỗi khi đánh giá: {e}")
```

### 4.2. Xử lý lỗi API
```python
try:
    thong_tin_thieu = evaluator.find_missing_information(cau_hoi, cau_tra_loi, tai_lieu)
except Exception as e:
    if "API" in str(e):
        print("Lỗi kết nối API. Vui lòng kiểm tra lại API key và kết nối mạng.")
    else:
        print(f"Lỗi không xác định: {e}")
```

## 5. Cấu hình Nâng cao

### 5.1. Tùy chỉnh Prompt đánh giá
```python
custom_prompt = """Bạn là chuyên gia đánh giá hệ thống RAG.
Hãy kiểm tra xem câu trả lời có chính xác dựa trên ngữ cảnh không.
Cho điểm từ 0 đến 100.
Giải thích điểm số của bạn.
Tìm ra các vấn đề hoặc thông tin bị thiếu."""

evaluator = RAGEvaluator(evaluation_prompt=custom_prompt)
```

## 6. Best Practices

1. **Chọn Model phù hợp**
   - GPT-4: Cho đánh giá chi tiết và chính xác
   - GPT-3.5: Cho đánh giá nhanh và tiết kiệm

2. **Điều chỉnh Temperature**
   - 0.0: Đánh giá nhất quán
   - 0.3: Cho phép một chút linh hoạt trong đánh giá

3. **Quản lý Token**
   - Giới hạn max_tokens để kiểm soát chi phí
   - Tối ưu hóa nội dung tài liệu

4. **Xử lý Kết quả**
   - Luôn kiểm tra điểm số
   - Phân tích phản hồi chi tiết
   - Theo dõi thông tin thiếu

5. **Xử lý Lỗi**
   - Luôn sử dụng try-catch
   - Kiểm tra kết nối API
   - Xác thực API key