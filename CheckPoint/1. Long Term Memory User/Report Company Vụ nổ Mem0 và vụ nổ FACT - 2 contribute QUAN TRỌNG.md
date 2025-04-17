
  

  

# giải pháp:

- các kỹ thuật cải thiện khả năng Chunking, Retrieval Hội thoại.
    

  

  

- Định nghĩa. Kỹ thuật : summary, ... Định nghĩa phiên, định nghĩa hội thoại, ... => summary, ... CÓ KỸ THUẬT SUMMARY, kỹ thuật đó như nào? sâu hơn đi? Summary hội thoại nó khác summary văn bản.
    

  

1. MOTIVATION RÕ RÀNG => MỚI TRIỂN?
    
2. DATASET? CÁCH HỌ ĐÁNH GIÁ NHƯ NÀO?
    
3. METHOD? KỸ THUẬT ĐÓ LÀ GÌ? ? KỸ THUẬT ĐÓ NHƯ NÀO? => ĐỊNH NGHĨA ĐƯỢC CÁC KHÁI NIỆM => MỚI CÓ KỸ THUẬT.
    

- CÁC KỸ THUẬT SUMMARY NHƯ NÀO?
    
- ...
    
- ...
    

4. EXTRACT ?
    

---

# **Đề tài**: **"Long-Term Memory Augmentation for Conversational Question Answering Systems"** _(Tăng cường trí nhớ dài hạn cho các hệ thống hỏi đáp hội thoại)_

1. # Motivation
    

### 🔥 **Động lực chính:**

1. **Cá nhân hóa sâu sắc** Ghi nhớ thông tin, sở thích và hành vi người dùng giúp chatbot/robot phản hồi phù hợp, tạo cảm giác thân quen và tăng sự hài lòng. 👉 **Ví dụ**: Một người dùng thường tìm công thức ăn chay. Chatbot ghi nhớ điều này và luôn gợi ý món chay, thay vì các công thức ngẫu nhiên. 👉 _Nếu không có điều này_: Chatbot có thể liên tục đề xuất món mặn, gây phiền toái và làm người dùng mất thiện cảm, từ đó giảm khả năng quay lại sử dụng.
    
2. **Duy trì và tái sử dụng ngữ cảnh** Trí nhớ dài hạn cho phép hiểu được lịch sử hội thoại qua nhiều phiên làm việc, giữ mạch logic và tránh yêu cầu người dùng phải lặp lại thông tin. 👉 **Ví dụ**: Trong một chuỗi hội thoại về đơn hàng, chatbot nhớ rằng người dùng đang khiếu nại về sản phẩm X từ phiên trước và tiếp tục hỗ trợ ngay ở phiên sau. 👉 _Nếu không có điều này_: Người dùng sẽ phải lặp lại toàn bộ thông tin khi quay lại, gây bực bội và tạo cảm giác chatbot thiếu chuyên nghiệp.
    
3. **Hỗ trợ tác vụ dài hạn** Các ứng dụng như theo dõi học tập, chăm sóc sức khỏe hoặc hành trình khách hàng yêu cầu chatbot ghi nhớ và cập nhật tiến trình liên tục. 👉 **Ví dụ**: Một chatbot học tập ghi nhớ rằng học sinh đã yếu ở phần thì quá khứ hoàn thành và tiếp tục luyện tập điểm này trong các buổi sau. 👉 _Nếu không có điều này_: Chatbot sẽ lặp lại những nội dung học đã ổn, bỏ sót điểm yếu của học sinh, làm giảm hiệu quả học tập và cảm giác “được đồng hành” của người học.
    

---

## 1.1 Dẫn chứng

![](https://csg2ej4iz2hz.sg.larksuite.com/space/api/box/stream/download/asynccode/?code=MDE5ZDE4NDZlNzk0NjEyMzJhZmQyN2FhMWU0ZjNmMjhfR1h3UVpiaFBOY0twYVI3WE1CNFd1S0VqdGdDajF1T2lfVG9rZW46RjJoRGJiRGl0b1ZDd3R4aXk2cmxqRW1yZ05iXzE3NDQ5MDA2MDk6MTc0NDkwNDIwOV9WNA)

Memory of Personalization

---

  

  

2. # Datasets và các metrics
    

![](https://csg2ej4iz2hz.sg.larksuite.com/space/api/box/stream/download/asynccode/?code=MWViMzFmOWJmYTEzYjA0ZDkzOTRiYWQwMWJhNTZmYTZfaHZmWExyd1RWVmpMOUE4VmsyZDNzV3ltUE1ubVJqMGtfVG9rZW46WGlNamJQd2p2b05MbHR4dk1sYWwzS0NKZ3hlXzE3NDQ5MDA2MDk6MTc0NDkwNDIwOV9WNA)

  

LongMemEval là một bộ dữ liệu toàn diện, được thiết kế để đánh giá khả năng ghi nhớ dài hạn của các trợ lý trò chuyện. Bộ dữ liệu này bao gồm 500 câu hỏi chất lượng cao, tập trung vào năm khả năng cốt lõi:

  

1. **Trích xuất thông tin (Information Extraction):** Khả năng nhớ lại thông tin cụ thể từ lịch sử tương tác dài, bao gồm cả chi tiết do người dùng hoặc trợ lý cung cấp.
    
2. **Lý luận đa phiên (Multi-Session Reasoning):** Khả năng tổng hợp thông tin từ nhiều phiên trò chuyện để trả lời các câu hỏi phức tạp yêu cầu sự tổng hợp và so sánh.
    
3. **Cập nhật kiến thức (Knowledge Updates):** Khả năng nhận biết và cập nhật thông tin cá nhân của người dùng theo thời gian.
    
4. **Lý luận thời gian (Temporal Reasoning):** Nhận thức về các khía cạnh thời gian của thông tin người dùng, bao gồm cả thời gian được đề cập rõ ràng và siêu dữ liệu thời gian trong các tương tác.
    
5. **Từ chối trả lời (Abstention):** Khả năng từ chối trả lời các câu hỏi liên quan đến thông tin không được đề cập trong lịch sử tương tác.
    

  

  

---

3. # 1 cách làm đơn giản nhất thủy tổ của các bài Memory:
    

https://arxiv.org/pdf/2410.10813

  

![](https://csg2ej4iz2hz.sg.larksuite.com/space/api/box/stream/download/asynccode/?code=NWYzYTE1NmJmMzYwMjcxN2FjMTA5NjBjZWI0ZDc1ZjdfMDVQSVNPU2VacnppOHZ6ZGZ6UDNwaEUzU0I1c0ZXU0RfVG9rZW46TlBhcGJmajNnb0Y0QXR4QmZxQWxySmc5Z0JlXzE3NDQ5MDA2MDk6MTc0NDkwNDIwOV9WNA)

  

![](https://csg2ej4iz2hz.sg.larksuite.com/space/api/box/stream/download/asynccode/?code=ODllYTExNDRiZWJkZGIwYjA4OTE5N2NjMGM2NzY3MGVfR0pqRVNaUXAyQU9UR0ZtNTNUWmdDTThRNGtsQlFwNzNfVG9rZW46TXl3QWJSNVM5b3NsemR4d08wdWw4ME1MZ3FiXzE3NDQ5MDA2MDk6MTc0NDkwNDIwOV9WNA)

  

- Llam3. 1- 8b
    
- Dataset cũ: 2023
    

```c++
Session
  ↓
Conversation-aware Chunking (LLMs Chunk + Raptor Chunk)
  ↓
Chunk-level Value → Extract (summary, fact, keyphrase)
  ↓
Embed:
   - K1 = V + summary
   - K2 = V + fact
   - K3 = V + keyphrase
  ↓
Phase 1: Coarse Retrieval từ summary/keyphrase
  ↓
Phase 2: Fine Retrieval từ facts
  ↓
Reading Strategy: CoN + JSON (Chain-of-Note)
  ↓
Answer
```

```c++
2. Về cách làm đơn giản nhất được xử lý rất giống RAG 
- Xử lý trước: 1 dialog thì gồm nhiều session, 1 session thì gồm nhiều round. (1 round là 1 lượt mà User trò chuyện với AI). 
Đem 1 round/1 session => Chunking ra => Đi embedding. 
- Khi realtime: Khi user có câu hỏi đến thì sẽ embedding câu hỏi + Query nó trong tập Embedding DB để tìm ra TOP K embedding gần nhất. 
=> Lấy làm context. Câu hỏi + Context => sinh ra câu trả lời. 
```

  

---

4. # Các giải pháp hiện đại với Graph:
    

LangGraph

**LangGraph** là một **framework** được giới thiệu trong khóa học "Long-Term Agentic Memory with LangGraph" do Harrison Chase, Co-Founder và CEO của LangChain, giảng dạy. Khóa học này hướng dẫn cách xây dựng một **agent** với khả năng **ghi nhớ dài hạn**, cụ thể là trong việc quản lý email cá nhân.

  

**Điểm mới mà LangGraph đề cập đến**:

  

1. **Tích hợp ba loại memory trong agent**:
    
    2. **Semantic Memory**: Lưu trữ các **facts** về người dùng, như sở thích, thói quen, để sử dụng trong các tương tác sau này.
        
    3. **Episodic Memory**: Ghi nhớ các **tình huống cụ thể** đã xảy ra trong quá khứ, giúp agent hiểu ngữ cảnh và cải thiện phản hồi.
        
    4. **Procedural Memory**: Lưu trữ các **hướng dẫn và quy trình** mà agent cần tuân theo, giúp tối ưu hóa hành vi dựa trên phản hồi. ??
        

  

---

```c++
[Input hội thoại hiện tại]

       ↓  
[Updater] — cập nhật memory

       ↓  
[Retriever] — truy xuất memory

       ↓  
[LLM Core] — sinh phản hồi + cá nhân hóa

       ↓  
[Output phản hồi]
```

```c++
Session
  ↓
Conversation-aware Chunking (LLMs Chunk + Raptor Chunk)
  ↓
Chunk-level Value → Extract (summary, fact, keyphrase)
  ↓
Embed:
   - K1 = V + summary
   - K2 = V + fact
   - K3 = V + keyphrase
  ↓
Phase 1: Coarse Retrieval từ summary/keyphrase
  ↓
Phase 2: Fine Retrieval từ facts
  ↓
Reading Strategy: CoN + JSON (Chain-of-Note)
  ↓
Answer
```

  

## 2.2 Mem0 - Graph Memory

  

  

## 2.3 Kiến trúc Zep - Graph -2-2025

  

Zep sử dụng **Graphiti** – một công cụ xây dựng đồ thị tri thức có tính thời gian (temporal), chia dữ liệu thành ba lớp:

  

1. **Episode Subgraph** (các “episode”):
    
    2. Mỗi “episode” là một đơn vị dữ liệu thô, ví dụ: một tin nhắn, một đoạn text, hay một JSON.
        
    3. Thông tin ở đây mang tính “episodic” – giống khái niệm “bộ nhớ sự kiện” (episodic memory) trong tâm lý học.
        
2. **Semantic Entity Subgraph** (các thực thể và quan hệ):
    
      
    
    2. Từ những “episode” (tin nhắn), hệ thống trích xuất **thực thể** (entity) và **mối quan hệ** (fact/edge) giữa các thực thể đó (ví dụ: “A làm việc tại công ty B từ năm 2020 đến 2022”).
        
    3. Thêm bước “entity resolution” để gộp những thực thể trùng nhau (nhưng có thể được nhắc với tên hơi khác) thành một node duy nhất.
        
    4. Thông tin thời gian (thời điểm có hiệu lực, lúc bắt đầu/kết thúc) được lưu cùng các quan hệ để theo dõi thay đổi của sự thật (“fact”) theo thời gian.
        
3. **Community Subgraph** (các cụm cộng đồng):
    
    2. Hệ thống gom nhóm các thực thể, mối quan hệ lại thành “cộng đồng” (community) – về cơ bản là các cụm các node liên quan chặt chẽ.
        
    3. Mỗi cộng đồng có một “summary” (tóm tắt) mô tả khái quát các thực thể và thông tin bên trong.
        

  

Với cấu trúc này, Zep duy trì được lịch sử thay đổi của thông tin, vừa có thể linh hoạt trả lời các câu hỏi mang tính thời gian (ví dụ: “A đã làm việc ở đâu vào năm ngoái?”) vừa có thể tóm tắt/khái quát bằng các summary.

  

---

  

## 3. Cơ chế lưu trữ và truy xuất (Memory Retrieval)

  

Zep áp dụng chiến lược tìm kiếm (search) nhiều bước:

  

1. **Tạo index (xây dựng embeddings, BM25, v.v.)**:
    
    2. Mỗi “fact” và mỗi “thực thể” đều có embedding (vector) để phục vụ tìm kiếm theo độ tương đồng cosine.
        
    3. Ngoài ra, tên hoặc mô tả (summary) của thực thể, fact có thể dùng cho tìm kiếm full-text (BM25).
        
    4. Có thể sử dụng tìm kiếm theo khoảng cách trên đồ thị (breadth-first search) để truy xuất thông tin gần các nút quan trọng.
        
2. **Kết hợp, xếp hạng lại (reranker)**:
    
    5. Sau khi lấy được danh sách kết quả theo nhiều cách (cosine, BM25, BFS), Zep áp dụng các thuật toán như RRF (Reciprocal Rank Fusion), MMR (Maximal Marginal Relevance), hoặc mô hình cross-encoder để xếp hạng lại, ưu tiên các kết quả quan trọng nhất.
        
3. **Kết hợp dữ liệu thành “context string”**:
    
    2. Từ top-N “edges” (fact) và “nodes” (entity) được xếp hạng cao, Zep xây dựng một đoạn văn bản (context) có cấu trúc, để đưa vào prompt cho LLM. Đoạn này thường bao gồm:
        
        - Các fact kèm thời gian hiệu lực
            
        - Thông tin tóm tắt về thực thể
            
        - Tóm lược cộng đồng nếu cần
            

  

Cách làm này giúp “nối dài” bộ nhớ của LLM mà không cần tải toàn bộ lịch sử vào ngữ cảnh. Hơn nữa, do Zep có khả năng ghi nhận và vô hiệu hóa (invalidate) các fact cũ khi có mâu thuẫn mới, nó cho phép cập nhật thông tin động khá hiệu quả.