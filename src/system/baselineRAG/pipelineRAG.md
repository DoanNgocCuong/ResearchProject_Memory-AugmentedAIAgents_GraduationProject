```bash
(.venv) PS D:\GIT\ResearchProject_Memory-AugmentedAIAgents_GraduationProject\src\system\baselineRAG\layers> python .\pipelineRAG.py
Đang load .env từ: D:\GIT\ResearchProject_Memory-AugmentedAIAgents_GraduationProject\src\system\baselineRAG\layers\_04_retrieval\.env
Đang load .env từ: D:\GIT\ResearchProject_Memory-AugmentedAIAgents_GraduationProject\src\system\baselineRAG\layers\_04_retrieval\.env
🚀 Testing RAG Pipeline...

==================================================
TEST 1: BM25 Retriever + GPT-4o-mini
==================================================
RAG Pipeline initialized with:
  - Retriever: bm25
  - Vector Store: qdrant
  - Generator: gpt-4o-mini
  - K documents: 3
  - Documents loaded: 4

🔍 Retrieving documents for: 'RAG là gì?'
✅ Found 3 relevant documents
📄 Retrieved documents preview:
   1. RAG (Retrieval-Augmented Generation) là một kỹ thuật kết hợp việc tìm kiếm thông tin liên quan với v...
   2. Python là ngôn ngữ lập trình phổ biến trong AI và machine learning. Các thư viện như LangChain giúp ...
🤖 Generating answer...
✅ Answer generated successfully

📋 Question: RAG là gì?
💡 Answer: RAG (Retrieval-Augmented Generation) là một kỹ thuật kết hợp việc tìm kiếm thông tin liên quan với việc tạo sinh ngôn ngữ, giúp cải thiện độ chính xác của các mô hình ngôn ngữ lớn.
📚 Sources: ['rag_intro.txt', 'gpt4_info.txt', 'python_ai.txt']
📄 Documents used: 3

==================================================
TEST 2: Vector Retriever
==================================================
📝 Attempting to index 4 documents...
⚠️  Warning: Could not verify/index documents: 'QdrantStore' object has no attribute 'add_doccuments'
    This might be okay if documents are already indexed
RAG Pipeline initialized with:
  - Retriever: vector
  - Vector Store: qdrant
  - Generator: gpt-4o-mini
  - K documents: 2
  - Documents loaded: 4

🔍 Retrieving documents for: 'Vector search hoạt động như thế nào?'
Error in similarity search: Unexpected Response: 404 (Not Found)
Raw response content:
b'{"status":{"error":"Not found: Collection `test_rag_pipeline` doesn\'t exist!"},"time":8.247e-6}'
✅ Found 0 relevant documents
⚠️  No documents found, trying fallback...
📚 Using fallback documents: 2
🤖 Generating answer...
✅ Answer generated successfully

📋 Question: Vector search hoạt động như thế nào?
💡 Answer: Vector search hoạt động bằng cách sử dụng embeddings, tức là các đại diện số cho văn bản, để tìm kiếm tài liệu tương tự về mặt ngữ nghĩa. Khi một truy vấn được thực hiện, nó cũng được chuyển đổi thành một embedding. Hệ thống sau đó so sánh embedding của truy vấn với các embedding của tài liệu trong cơ sở dữ liệu để xác định tài liệu nào có ngữ nghĩa gần gũi nhất. Phương pháp này thường hiệu quả hơn so với tìm kiếm theo từ khóa trong nhiều trường hợp.
📚 Sources: ['rag_intro.txt', 'vector_search.txt']

==================================================
TEST 3: Batch Queries
==================================================

📝 Processing question 1/2

🔍 Retrieving documents for: 'Python có ưu điểm gì trong AI?'
✅ Found 3 relevant documents
📄 Retrieved documents preview:
   1. Python là ngôn ngữ lập trình phổ biến trong AI và machine learning. Các thư viện như LangChain giúp ...
   2. OpenAI GPT-4 là một mô hình ngôn ngữ lớn có khả năng hiểu và tạo sinh văn bản một cách tự nhiên. Nó ...
🤖 Generating answer...
✅ Answer generated successfully

📝 Processing question 2/2

🔍 Retrieving documents for: 'GPT-4 là gì?'
✅ Found 3 relevant documents
📄 Retrieved documents preview:
   1. OpenAI GPT-4 là một mô hình ngôn ngữ lớn có khả năng hiểu và tạo sinh văn bản một cách tự nhiên. Nó ...
   2. Python là ngôn ngữ lập trình phổ biến trong AI và machine learning. Các thư viện như LangChain giúp ...
🤖 Generating answer...
✅ Answer generated successfully

1. Python có ưu điểm gì trong AI?
   Answer: Python có ưu điểm trong AI nhờ vào tính phổ biến của nó, cùng với các thư viện như LangChain giúp xâ...

2. GPT-4 là gì?
   Answer: GPT-4 là một mô hình ngôn ngữ lớn có khả năng hiểu và tạo sinh văn bản một cách tự nhiên, được sử dụ...

==================================================
TEST 4: System Information
==================================================
🔧 System Configuration:
   retriever_type: bm25
   vector_store_type: qdrant
   generator_model: gpt-4o-mini
   k_documents: 3
   temperature: 0
   collection_name: VIMQA_dev
   total_documents: 4

==================================================
TEST 5: Simple Functional Test
==================================================
RAG Pipeline initialized with:
  - Retriever: bm25
  - Vector Store: qdrant
  - Generator: gpt-4o-mini
  - K documents: 5
  - Documents loaded: 4

🔍 Retrieving documents for: 'Python có gì hay?'
✅ Found 4 relevant documents
   2. OpenAI GPT-4 là một mô hình ngôn ngữ lớn có khả năng hiểu và tạo sinh văn bản một cách tự nhiên. Nó ...
iên. Nó ...
🤖 Generating answer...
🤖 Generating answer...
✅ Answer generated successfully
✅ Answer generated successfully


📋 Question: Python có gì hay?
💡 Answer: Python là ngôn ngữ lập trình phổ biến trong AI và machine learning, với các thư viện nh📋 Question: Python có gì hay?
💡 Answer: Python là ngôn ngữ lập trình phổ biến trong AI và machine learning, với các thư viện nh💡 Answer: Python là ngôn ngữ lập trình phổ biến trong AI và machine learning, với các thư viện như LangChain giúp xây dựng ứng dụng AI dễ dàng hơn. Nó cũng hỗ trợ nhiều kỹ thuật tiên tiến như RAGư LangChain giúp xây dựng ứng dụng AI dễ dàng hơn. Nó cũng hỗ trợ nhiều kỹ thuật tiên tiến như RAG (Retrieval-Augmented Generation) để cải thiện độ chính xác trong việc tạo sinh ngôn ngữ.
📄 Documents used: 4


📑 Retrieved documents:
   1. Python là ngôn ngữ lập trình phổ biến trong AI và ...
   1. Python là ngôn ngữ lập trình phổ biến trong AI và ...
   1. Python là ngôn ngữ lập trình phổ biến trong AI và ...
   1. Python là ngôn ngữ lập trình phổ biến trong AI và ...
   2. OpenAI GPT-4 là một mô hình ngôn ngữ lớn có khả nă...
   3. Vector search sử dụng embeddings để tìm kiếm tài l...
   4. RAG (Retrieval-Augmented Generation) là một kỹ thu...

✅ RAG Pipeline testing completed!
(.venv) PS D:\GIT\ResearchProject_Memory-AugmentedAIAgents_GraduationProject\src\system\baselineRAG\layers>













```