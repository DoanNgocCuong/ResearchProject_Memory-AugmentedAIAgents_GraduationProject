# RAG Answer Generator Documentation

## 1. Cài đặt và Khởi tạo

### 1.1. Cài đặt thư viện
```python
from typing import List, Dict, Any
from langchain_core.documents import Document
from openai import OpenAI
import httpx
from dotenv import load_dotenv
import os

# Load biến môi trường
load_dotenv()
```

### 1.2. Khởi tạo Generator
```python
# Khởi tạo với cấu hình mặc định
generator = AnswerGenerator()

# Hoặc khởi tạo với cấu hình tùy chỉnh
generator = AnswerGenerator(
    model_name="gpt-4",        # Tên model OpenAI
    temperature=0.0,           # Độ sáng tạo (0.0 - 1.0)
    max_tokens=4096,          # Số token tối đa
    system_prompt="..."       # Prompt tùy chỉnh
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

### 2.2. Chuẩn bị câu hỏi
```python
cau_hoi = "RAG là gì?"
```

## 3. Sử dụng Generator

### 3.1. Tạo câu trả lời cơ bản
```python
# Tạo câu trả lời đơn giản
cau_tra_loi = generator.generate_answer(
    question=cau_hoi,
    documents=tai_lieu
)
print(f"Câu trả lời: {cau_tra_loi}")
```

### 3.2. Tạo câu trả lời với nguồn tham khảo
```python
# Tạo câu trả lời kèm nguồn
ket_qua = generator.generate_answer_with_sources(
    question=cau_hoi,
    documents=tai_lieu
)

# In kết quả
print(f"Câu trả lời: {ket_qua['answer']}")
print(f"Nguồn tham khảo: {ket_qua['sources']}")
```

## 4. Xử lý Lỗi

### 4.1. Try-Catch cơ bản
```python
try:
    cau_tra_loi = generator.generate_answer(cau_hoi, tai_lieu)
except Exception as e:
    print(f"Lỗi khi tạo câu trả lời: {e}")
```

### 4.2. Xử lý lỗi API
```python
try:
    ket_qua = generator.generate_answer_with_sources(cau_hoi, tai_lieu)
except Exception as e:
    if "API" in str(e):
        print("Lỗi kết nối API. Vui lòng kiểm tra lại API key và kết nối mạng.")
    else:
        print(f"Lỗi không xác định: {e}")
```

## 5. Cấu hình Nâng cao

### 5.1. Cấu hình Proxy
```python
# Thêm proxy vào file .env
# OPENAI_PROXY="http://your-proxy:port"

# Hoặc cấu hình trực tiếp
os.environ["OPENAI_PROXY"] = "http://your-proxy:port"
```

### 5.2. Tùy chỉnh System Prompt
```python
custom_prompt = """Bạn là trợ lý AI hữu ích. 
Hãy trả lời câu hỏi dựa trên ngữ cảnh được cung cấp.
Nếu không biết, hãy nói rằng bạn không biết.
Chỉ sử dụng thông tin từ ngữ cảnh để trả lời.
Giữ câu trả lời rõ ràng và đơn giản."""

generator = AnswerGenerator(system_prompt=custom_prompt)
```

## 6. Best Practices

1. **Chọn Model phù hợp**
   - GPT-4: Cho câu trả lời chất lượng cao
   - GPT-3.5: Cho câu trả lời nhanh và tiết kiệm

2. **Điều chỉnh Temperature**
   - 0.0: Câu trả lời nhất quán
   - 0.7: Câu trả lời sáng tạo hơn

3. **Quản lý Token**
   - Giới hạn max_tokens để kiểm soát chi phí
   - Tối ưu hóa nội dung tài liệu

4. **Xử lý Nguồn**
   - Luôn sử dụng generate_answer_with_sources khi cần tham khảo
   - Kiểm tra metadata của tài liệu

5. **Xử lý Lỗi**
   - Luôn sử dụng try-catch
   - Kiểm tra kết nối API
   - Xác thực API key 