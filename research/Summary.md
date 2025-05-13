

---
Báo cáo Tóm tắt: HippoRAG – Một Giải pháp Bộ nhớ Dài hạn cho LLM
1. Motivation: Tại sao cần HippoRAG?
Vấn đề của RAG truyền thống
- RAG truyền thống hoạt động bằng cách:
  - Chia văn bản thành các đoạn nhỏ (chunks).
  - Dùng các vector embedding lưu vào cơ sở dữ liệu vector (VectorDB).
  - Khi có truy vấn, tìm các vector có độ tương đồng cao, ném toàn bộ các đoạn văn đó vào context window của LLM.
- Hạn chế:
  - Multi-hop reasoning: Nếu thông tin cần thiết nằm rải rác ở nhiều đoạn văn, mỗi đoạn chỉ có mức độ tương đồng vừa phải, thì chúng dễ bị loại ra khỏi Top-K nên LLM không có đủ “nguyên liệu” để suy luận.
  - Compute và Latency: Các mô hình LLM hiện đại như GPT-4 Turbo hay Claude Opus có thể xử lý context rất dài (như 128k tokens) nhưng chi phí tính toán cao, làm tăng thời gian phản hồi và không thể scale cho các ứng dụng thực tế.
Giải pháp của HippoRAG
- HippoRAG (và HippoRAG 2) được xây dựng dựa trên cảm hứng từ bộ nhớ dài hạn của con người:
  - Neocortex (LLM): Xử lý và suy luận.
  - Hippocampus (Knowledge Graph – KG): Lưu trữ các mối liên kết tri thức.
  - Parahippocampal region: Kết nối và “gợi nhớ” các khái niệm liên quan.
- Kỹ thuật then chốt: Sử dụng thuật toán Personalized PageRank (PPR) trên KG để "lan tỏa" thông tin từ các seed node, qua đó thực hiện multi-hop reasoning ngay trong quá trình retrieval.
- Lợi ích:
  - Nâng cao khả năng truy xuất các mảnh thông tin liên kết (associativity) và hiểu ngữ cảnh (sense-making)
  - Giảm chi phí tính toán so với việc ném toàn bộ dữ liệu vào LLM
  - Tạo ra hệ thống có tính minh bạch (xAI) với khả năng ghi lại đường reasoning.

---
2. Dataset & Đánh giá
Các loại nhiệm vụ đánh giá
- Factual QA: Kiểm tra khả năng truy xuất thông tin cụ thể, ví dụ: NaturalQuestions (NQ), PopQA.
- Multi-hop QA (Associative Tasks): Yêu cầu kết nối nhiều mảnh thông tin rời rạc, ví dụ: 2WikiMultiHopQA, HotpotQA.
- Sense-making Tasks (Discourse Understanding): Đánh giá khả năng hiểu và tổng hợp ngữ cảnh phức tạp, ví dụ: NarrativeQA.
Chỉ số đánh giá
- Retrieval: Sử dụng các chỉ số như Recall@K để đo lường khả năng truy xuất các đoạn văn cần thiết.
- QA Performance: Đo lường qua Exact Match (EM) và F1 Score khi LLM sinh ra câu trả lời cuối cùng.
Ví dụ minh họa:
 Trong một tác vụ multi-hop, nếu cần hai đoạn văn (A và B) để reasoning mà RAG truyền thống chỉ retrieve được các đoạn văn rời rạc, LLM sẽ không thể nối lại logic. HippoRAG thông qua việc sử dụng KG và PPR đã “xâu chuỗi” các mảnh thông tin lại với nhau, đảm bảo rằng các đoạn văn cần thiết được xếp hạng cao và đưa vào context.

---
3. Method – Kỹ thuật và Các Khái niệm Cốt lõi
a. Offline Indexing – Xây dựng bộ nhớ
1. OpenIE by LLM:
  - Dùng LLM (ví dụ: Llama-3.3-70B-Instruct) để trích xuất các triple dạng (subject, predicate, object) từ từng đoạn văn. - **Predicate** chính là nhãn (label) của cạnh (edge) kết nối giữa hai nút đó.
  - Ví dụ: "Marie Curie won two Nobel Prizes" → ("Marie Curie", "won", "Nobel Prizes"). (Có hướng)
2. Knowledge Graph (KG) & Synonym Detection:
  - Các triple được kết hợp thành một KG, trong đó:
    - Phrase Nodes (sparse): Lưu trữ các khái niệm rút gọn.
    - Passage Nodes (dense): Lưu trữ toàn bộ đoạn văn gốc.
  - Context Edges: Nối các Passage Node với các Phrase Node được trích xuất từ chính đoạn đó.
  - Synonym Edges: Nối các phrase có ý nghĩa tương tự bằng cách so sánh cosine similarity của embedding.
3. Dense-Sparse Integration:
  - Sự kết hợp giữa sparse coding (phrase nodes – ý chính) và dense coding (passage nodes – ngữ cảnh chi tiết) giúp hệ thống vừa nhanh trong reasoning vừa chính xác về thông tin chi tiết.
b. Online Retrieval & QA – Từ truy vấn đến câu trả lời
1. Retrieval của Passages & Triples:
  - Sử dụng retriever (NV-Embed-v2) để lấy các đoạn văn và triple liên quan đến truy vấn.
2. Recognition Memory – Triple Filtering:
  - Dùng LLM kiểm tra lại các triple đã retrieve, loại bỏ các triple không thực sự liên quan.
3. Assigning Seed Node Weights:
  - Seed Nodes: Các node liên quan (có thể là phrase node hay passage node) được chọn làm điểm khởi đầu cho quá trình PPR.
  - Gán trọng số:
    - Phrase Nodes được gán weight factor 1.0 (giữ nguyên giá trị similarity).
    - Passage Nodes được gán weight factor thấp hơn (ví dụ: 0.05) nhằm kiểm soát sự lan tỏa của thông tin ngữ cảnh.
  - Ví dụ cụ thể:
    - Nếu một Phrase Node có similarity 0.72 → Effective weight = 0.72.
    - Một Passage Node với similarity 0.80 → Effective weight = 0.80 × 0.05 = 0.04.
3. Personalized PageRank (PPR) Graph Search:
- Một bước PPR = **multi-hop reasoning một lần**: tín hiệu lan từ seed → qua quan hệ → trồi lên ở passage liên quan dù xa 2-3 cạnh.
  - Thuật toán PPR "lan tỏa" xác suất từ các seed node qua các liên kết trong KG.
  - Việc này không chỉ re-rank các node ban đầu mà còn kết nối các mối quan hệ đa bước (multi-hop), giúp tìm ra các đoạn văn có liên kết logic tốt nhất với truy vấn.
  - Ví dụ minh họa: Nếu các Phrase Nodes có trọng số cao liên kết mạnh với một Passage Node, PPR sẽ “nâng” trọng số của Passage Node đó, cho phép nó được xếp hạng cao.
5. QA Reading:
  - Các đoạn văn được chọn qua quá trình PPR sẽ được đưa vào LLM để sinh ra câu trả lời cuối cùng.
c. Tóm tắt mối liên hệ giữa các khái niệm
- Triple: Ghi chú ngắn gọn từ văn bản, chứa thông tin dạng (subject, predicate, object).
- Seed Node: Các điểm khởi đầu trong KG được chọn dựa trên mức độ liên quan với truy vấn, được gán trọng số riêng biệt để ưu tiên thông tin quan trọng.
- PPR: Thuật toán giúp “traverse” KG, lan tỏa trọng số từ các seed node và re-rank các node liên quan theo cấu trúc liên kết của kiến thức, hỗ trợ multi-hop reasoning.

---
KẾT LUẬN
HippoRAG (và phiên bản nâng cấp HippoRAG 2) mang lại một giải pháp toàn diện cho các bài toán long-term memory của LLM bằng cách:
- Vượt qua hạn chế của RAG truyền thống: Không chỉ dựa vào vector retrieval đơn giản, mà còn kết nối thông tin qua một Knowledge Graph được xây dựng thông minh.
- Đáp ứng các yêu cầu của multi-hop reasoning: Thông qua thuật toán Personalized PageRank, hệ thống có thể “reason” qua nhiều bước để tìm ra các đoạn văn liên kết và trả lời chính xác.
- Tiết kiệm chi phí tính toán và tăng tính minh bạch: Không cần đưa toàn bộ dữ liệu vào LLM mỗi truy vấn, mà đã có quá trình retrieval có cấu trúc và khả năng ghi lại “đường reasoning” để dễ kiểm soát và debug.
Báo cáo này cung cấp một cái nhìn tổng quát và chi tiết về cách HippoRAG hoạt động, từ quá trình xây dựng bộ nhớ (offline indexing) đến việc truy xuất và reasoning trực tuyến (online retrieval & QA). Hy vọng bản báo cáo này sẽ giúp cô giáo hiểu rõ hơn về tầm quan trọng của việc kết hợp giữa vector retrieval và graph reasoning để xây dựng hệ thống AI thông minh, linh hoạt và hiệu quả.