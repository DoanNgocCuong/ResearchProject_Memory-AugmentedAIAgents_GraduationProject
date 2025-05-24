`compression_retriever.py` là một file trong module retrieval, được thiết kế để thực hiện tìm kiếm tài liệu dựa trên kỹ thuật nén ngữ cảnh (contextual compression). Cụ thể:

1. **Mục đích:**
   - Sử dụng LLM (Language Model) để nén/trích xuất thông tin quan trọng từ các tài liệu
   - Giúp tìm kiếm chính xác hơn bằng cách chỉ giữ lại thông tin liên quan đến câu hỏi

2. **Cách hoạt động:**
   ```python
   class CompressionRetriever(BaseRetriever):
       def __init__(self, vector_store: VectorStore, k: int = 4):
           # Khởi tạo LLM (GPT-3.5-turbo)
           self.llm = ChatOpenAI(model="gpt-3.5-turbo")
           
           # Tạo base retriever (vector search)
           self.base_retriever = VectorRetriever(vector_store, k=k)
           
           # Tạo compressor để nén tài liệu
           self.compressor = LLMChainExtractor.from_llm(self.llm)
           
           # Kết hợp compressor và retriever
           self.retriever = ContextualCompressionRetriever(
               base_compressor=self.compressor,
               base_retriever=self.base_retriever
           )
   ```

3. **Quy trình tìm kiếm:**
   - Bước 1: Tìm các tài liệu liên quan bằng vector search
   - Bước 2: Sử dụng LLM để nén/trích xuất thông tin quan trọng từ các tài liệu đó
   - Bước 3: Trả về các đoạn văn bản đã được nén, chỉ chứa thông tin liên quan

4. **Ví dụ:**
   ```python
   # Khởi tạo
   retriever = CompressionRetriever(vector_store, k=4)
   
   # Tìm kiếm
   results = retriever.retrieve_documents("What is vector database?")
   # Kết quả sẽ là các đoạn văn bản đã được nén, chỉ chứa thông tin về vector database
   ```

5. **Ưu điểm:**
   - Kết quả tìm kiếm chính xác hơn
   - Giảm nhiễu thông tin không liên quan
   - Tập trung vào thông tin quan trọng nhất

6. **Nhược điểm:**
   - Cần thêm thời gian xử lý do phải gọi LLM
   - Chi phí cao hơn do sử dụng LLM
   - Phụ thuộc vào chất lượng của LLM

Đây là một kỹ thuật nâng cao trong RAG (Retrieval-Augmented Generation), giúp cải thiện chất lượng tìm kiếm bằng cách kết hợp vector search với khả năng hiểu ngữ cảnh của LLM.


---
