# VẤN ĐỀ MÔ HÌNH EMBEDDING ĐANG GẶP TRÊN HUGGING FACE

Sau khi tìm hiểu các thông tin từ nhiều nguồn, tôi có thể xác định được nguyên nhân và giải pháp cho vấn đề các mô hình embedding đột ngột không hoạt động trong dự án của bạn.

## 1. VẤN ĐỀ

Dựa trên log lỗi, hệ thống của bạn đang gặp vấn đề với mã lỗi 401 (Unauthorized) khi cố gắng truy cập các mô hình embedding từ Hugging Face. Tất cả các mô hình embedding tiếng Việt được thử nghiệm đều trả về lỗi 401, mặc dù kết nối đến API Hugging Face thành công thông qua lệnh `whoami`.

## 2. NGUYÊN NHÂN

Sau khi nghiên cứu, tôi phát hiện có một số thay đổi quan trọng trong hệ thống API của Hugging Face trong những tháng gần đây:

### 2.1. Thay đổi cơ chế API của Hugging Face

Hugging Face đã chuyển từ mô hình "Serverless Inference API" cũ sang hệ thống mới là "Inference Providers" vào đầu năm 2025. Như được đề cập trong [tài liệu chính thức](https://huggingface.co/docs/inference-providers/en/index), đây là một cách tiếp cận hoàn toàn mới, làm thay đổi cách thức truy cập và xác thực các mô hình.

### 2.2. Thay đổi chính sách hạn ngạch và giới hạn sử dụng

Có nhiều người dùng báo cáo rằng Hugging Face đã thay đổi giới hạn API từ 1000 lệnh gọi miễn phí mỗi ngày xuống còn $0.10 tín dụng miễn phí mỗi tháng, ảnh hưởng đến nhiều người dùng, đặc biệt là trong các tình huống sử dụng không thương mại.

### 2.3. Lỗi kỹ thuật trên hệ thống

Như được thấy trong [bài thảo luận này](https://discuss.huggingface.co/t/inference-api-stopped-working/150492), bắt đầu từ giữa tháng 4/2025, nhiều người dùng báo cáo rằng API suy luận (Inference API) đột nhiên ngừng hoạt động đối với nhiều mô hình khác nhau, trả về lỗi 401 hoặc thông báo "Model xxx is not supported HF inference api".

### 2.4. Thay đổi cách xác thực token

Cách thức xác thực token đã thay đổi, yêu cầu thêm quyền "inference.serverless.write" cho các token mới, và có thể các token cũ không còn khả năng tương thích.

### 2.5. Giới hạn mô hình được hỗ trợ

Hugging Face đã giới hạn đáng kể số lượng mô hình được hỗ trợ thông qua API suy luận miễn phí. Thay vì hỗ trợ hàng chục nghìn mô hình như trước đây, giờ đây chỉ còn một số mô hình được chọn lọc.

## 3. GIẢI PHÁP

Dựa trên phân tích trên, dưới đây là các giải pháp tôi đề xuất:

### 3.1. Tạo token API mới với quyền cụ thể

```bash
# Truy cập vào trang https://huggingface.co/settings/tokens/new
# Tạo token mới với các quyền sau:
# - inference.serverless.write
```

Đảm bảo chọn token loại "fine-grained" thay vì token đơn giản, và chọn phạm vi "Make calls to Inference Providers" như được đề cập trong tài liệu mới.

### 3.2. Cập nhật cách truyền mới 

```bash
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="hf-inference",
    api_key="hf_xxxxxxxxxxxxxxxxxxxxxxxx",
)

result = client.sentence_similarity(
    inputs={
    "source_sentence": "That is a happy person",
    "sentences": [
        "That is a happy dog",
        "That is a very happy person",
        "Today is a sunny day"
    ]
},
    model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
)
```

## 4. IMPLEMENTATION FIX

### 4.1. Trước khi fix

Ban đầu, code sử dụng `HuggingFaceInferenceAPIEmbeddings` từ `langchain_community.embeddings`:

```python
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=Config.HUGGINGFACE_API_KEY,
    model_name=Config.EMBEDDINGS_MODEL_NAME
)

# Sử dụng
query_vector = embeddings.embed_query(query)
```

### 4.2. Sau khi fix

Đã chuyển sang sử dụng `InferenceClient` trực tiếp từ `huggingface_hub`:

```python
from huggingface_hub import InferenceClient

embeddings_client = InferenceClient(
    provider="hf-inference",
    api_key=Config.HUGGINGFACE_API_KEY
)

# Sử dụng
query_vector = embeddings_client.feature_extraction(
    model=Config.EMBEDDINGS_MODEL_NAME,
    text=query  # Lưu ý: sử dụng 'text' thay vì 'inputs'
)
```

### 4.3. Những thay đổi chính

1. Thay đổi import:
   - Từ: `from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings`
   - Thành: `from huggingface_hub import InferenceClient`

2. Thay đổi cách khởi tạo client:
   - Thêm tham số `provider="hf-inference"`
   - Đổi tên biến từ `embeddings` thành `embeddings_client`

3. Thay đổi cách gọi API:
   - Từ: `embeddings.embed_query(query)`
   - Thành: `embeddings_client.feature_extraction(model=..., text=query)`

4. Thay đổi tên tham số:
   - Sử dụng `text` thay vì `inputs` trong phương thức `feature_extraction()`

### 4.4. Lợi ích của cách triển khai mới

1. Truy cập trực tiếp API của Hugging Face, giảm độ trễ
2. Kiểm soát tốt hơn các tham số và xử lý lỗi
3. Tương thích với hệ thống Inference Providers mới
4. Dễ dàng mở rộng và tùy chỉnh theo nhu cầu



---

# VẤN ĐỀ MÔ HÌNH EMBEDDING ĐANG GẶP TRÊN HUGGING FACE

[Previous content remains the same...]

## 5. IMPLEMENTATION MỚI (KHÔNG SỬ DỤNG LANGCHAIN)

### 5.1. Cấu trúc mới

```python
from huggingface_hub import InferenceClient
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Khởi tạo client
client = InferenceClient(
    provider="hf-inference",
    api_key=HUGGINGFACE_API_KEY
)

# Hàm lấy embedding
def get_embedding(text):
    return client.feature_extraction(
        model=EMBEDDINGS_MODEL_NAME,
        text=text
    )
```

### 5.2. Những thay đổi chính

1. **Loại bỏ phụ thuộc vào LangChain**:
   - Không sử dụng `HuggingFaceInferenceAPIEmbeddings`
   - Truy cập trực tiếp Hugging Face API thông qua `InferenceClient`
   - Tương tác trực tiếp với Qdrant thông qua `QdrantClient`

2. **Cải thiện hiệu suất**:
   - Giảm độ trễ do loại bỏ lớp trung gian LangChain
   - Xử lý trực tiếp các vector và metadata
   - Kiểm soát tốt hơn quá trình tạo collection và upsert

3. **Xử lý lỗi tốt hơn**:
   - Kiểm tra và tạo collection tự động
   - Xử lý các trường hợp dữ liệu không đồng nhất
   - Log chi tiết quá trình xử lý

### 5.3. Cấu trúc dữ liệu

1. **Vector Database (Qdrant)**:
   ```python
   {
       "collection_name": "legal_rag",
       "vectors_config": {
           "size": 768,  # Kích thước vector từ model
           "distance": "Cosine"
       }
   }
   ```

2. **Document Structure**:
   ```python
   {
       "id": int,
       "vector": List[float],
       "payload": {
           "source": str,
           "question": str,
           "page_content": str
       }
   }
   ```

### 5.4. Lợi ích của cách triển khai mới

1. **Đơn giản hóa**:
   - Giảm số lượng dependencies
   - Code dễ đọc và bảo trì hơn
   - Ít lớp trừu tượng hơn

2. **Hiệu suất**:
   - Xử lý nhanh hơn do loại bỏ lớp trung gian
   - Kiểm soát tốt hơn việc tạo và quản lý vector
   - Tối ưu hóa việc sử dụng bộ nhớ

3. **Bảo mật**:
   - Kiểm soát tốt hơn việc xác thực API
   - Xử lý an toàn các thông tin nhạy cảm
   - Dễ dàng cập nhật và quản lý API keys

4. **Khả năng mở rộng**:
   - Dễ dàng thêm các tính năng mới
   - Tùy chỉnh quá trình xử lý dữ liệu
   - Tích hợp với các hệ thống khác

### 5.5. Hướng dẫn sử dụng

1. **Cài đặt dependencies**:
   ```bash
   pip install huggingface-hub qdrant-client pandas python-dotenv
   ```

2. **Cấu hình môi trường**:
   ```env
   HUGGINGFACE_API_KEY=your_api_key
   QDRANT_API_KEY=your_qdrant_key
   QDRANT_HOST=localhost
   QDRANT_PORT=6333
   EMBEDDINGS_MODEL_NAME=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
   COLLECTION_NAME=legal_rag
   ```

3. **Chạy script**:
   ```bash
   python create_vector_database.py --excel path/to/data.xlsx
   # hoặc
   python create_vector_database.py --text-dir path/to/text/files
   ```