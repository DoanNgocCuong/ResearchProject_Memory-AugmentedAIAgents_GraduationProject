Read 2 - GOOD - LONGMEMEVAL - Đơn giản hơn Zep Khá Baseline - Benchmarking Chat Assistants on Long-Term Interactive Memory - Dataset của bài Zep nên mình tìm được bài này

- https://arxiv.org/abs/2410.10813
    

---

  

```Plain
4. Thực nghiệm và đánh giá
4.1 Deep Memory Retrieval (DMR)
DMR (giới thiệu trong MemGPT) có 500 cuộc hội thoại nhiều phiên (multi-session).
Zep đạt 94.8% độ chính xác khi dùng GPT-4-turbo (và 98.2% khi dùng một biến thể GPT-4o-mini), nhỉnh hơn so với MemGPT (93.4%).
Tuy nhiên, bộ DMR chỉ có hội thoại khá ngắn (khoảng 60 tin nhắn mỗi cuộc), chưa thực sự kiểm tra khả năng “siêu dài hạn”.
4.2 LongMemEval (LME)
LongMemEval có các đoạn hội thoại dài hơn nhiều (trung bình 115.000 tokens), mô phỏng tình huống doanh nghiệp thực tế phức tạp.
Zep cải thiện kết quả so với baseline (dùng toàn bộ hội thoại) ở hầu hết các loại câu hỏi, đặc biệt:
Loại câu “multi-session,” “preference,” “temporal reasoning” tăng đáng kể.
Độ trễ (latency) giảm đến 90% so với việc nhét toàn bộ hội thoại vào prompt (vì prompt của Zep ngắn gọn hơn).

=====
Search các bài báo liên quan đến dataset này 
```

  

  

Dưới đây là tóm tắt nội dung bài báo “LONGMEMEVAL: Benchmarking Chat Assistants on Long-Term Interactive Memory.” Bài báo tập trung đề xuất một bộ dữ liệu và phương pháp đánh giá nhằm kiểm tra khả năng “ghi nhớ dài hạn” của các trợ lý hội thoại (chat assistant) dựa trên mô hình ngôn ngữ lớn (LLM).

  

---

1. # Tổng quan bài:
    

## 1. Động lực và bối cảnh

  

- **Vấn đề**: Các LLM ngày càng mạnh, hỗ trợ nhiều tác vụ hội thoại. Tuy nhiên, khi cần “cá nhân hoá” câu trả lời hoặc cần “ghi nhớ” nhiều phiên trò chuyện dài hạn với người dùng (vd: thông tin cá nhân, lịch sử hoạt động), các mô hình thường gặp khó khăn.
    
- **Khoảng trống**: Đã có một số nghiên cứu về RAG (Retrieval-Augmented Generation) hay memory-augmented chatbots, nhưng hầu hết đánh giá trên các bộ dữ liệu tương đối ngắn hoặc thiên về đối thoại người–người (thay vì người–máy). Chưa có tiêu chuẩn đánh giá toàn diện cho các năng lực “bộ nhớ dài hạn” trên kịch bản tương tác nhiều phiên thực tế.
    

  

---

  

## 2. Giới thiệu **LONGMEMEVAL**

  

**LONGMEMEVAL** là một bộ dữ liệu (benchmark) kiểm tra 5 khả năng cốt lõi của bộ nhớ dài hạn trong hệ thống trợ lý hội thoại:

  

1. **Information Extraction (IE)**: Trích xuất và “nhớ” lại chính xác các thông tin cụ thể xuất hiện rải rác trong hội thoại.
    
2. **Multi-Session Reasoning (MR)**: Tổng hợp thông tin từ nhiều phiên trò chuyện để trả lời câu hỏi phức tạp.
    
3. **Temporal Reasoning (TR)**: Suy luận về mặt thời gian, bao gồm thời điểm, thứ tự sự kiện.
    
4. **Knowledge Updates (KU)**: Nhận biết và cập nhật thông tin mới khi người dùng thay đổi dữ liệu cá nhân.
    
5. **Abstention (ABS)**: Từ chối trả lời khi thông tin không có sẵn trong lịch sử (hỏi về dữ liệu “chưa từng nhắc đến”).
    

  

### Cách xây dựng bộ dữ liệu

  

- Mỗi **câu hỏi (question)** kèm **phần trả lời chuẩn (golden answer)** được “giấu” trong một lịch sử hội thoại (hay “haystack”) rất dài (có thể tới 115 nghìn token trong **LONGMEMEVAL** phiên bản “S” hoặc 1.5 triệu token trong phiên bản “M”).
    
- Các **evidence sessions** (những đoạn hội thoại chứa dữ liệu then chốt) được chèn xen kẽ cùng nhiều đoạn hội thoại khác (vô thưởng vô phạt) để mô phỏng bối cảnh “needle in a haystack”.
    
- Các đoạn hội thoại được tạo ra nhờ cách kết hợp dữ liệu thực (chẳng hạn ShareGPT, UltraChat) và các đoạn tự mô phỏng (self-chat) giữa hai mô hình LLM (đã được human kiểm duyệt).
    
- Mỗi **câu hỏi** có thể yêu cầu nhớ thông tin từ 1 đến 6 session khác nhau trong lịch sử, có thể có thêm yếu tố thời gian (timestamp).
    

  

### Thống kê và độ khó

  

- Trung bình khoảng **115 nghìn tokens** (LONGMEMEVAL-S) hoặc **500 session ~ 1.5 triệu tokens** (LONGMEMEVAL-M) cho mỗi bài test.
    
- Bộ dữ liệu gồm 500 “bài test” (mỗi bài là một câu hỏi). Các câu hỏi đa dạng về loại (7 loại chính, tương ứng 5 năng lực cốt lõi).
    
- Thử nghiệm ban đầu cho thấy các mô hình lớn (ví dụ GPT-4, Llama 3.1) vẫn suy giảm độ chính xác đáng kể (từ 30%–60%) khi phải “đọc” toàn bộ lịch sử. Các dịch vụ thương mại (ChatGPT, Coze) cũng gặp khó.
    

  

---

  

## 3. Khung đánh giá và kết quả thí điểm

  

### 3.1 Thử nghiệm trên các hệ thống thương mại

  

- Tác giả tiến hành chat “từng bước” (session-by-session) với ChatGPT và Coze, rồi đặt câu hỏi cuối cùng. Kết quả:
    
    - **ChatGPT** (GPT-4o) và **Coze** (GPT-4o, GPT-3.5) đều bị giảm độ chính xác so với việc chỉ đơn giản “đọc offline” toàn bộ lịch sử một lần.
        
    - Dường như khi số session tăng, hai hệ thống có xu hướng ghi đè/làm mất mát thông tin cũ.
        

  

### 3.2 Thử nghiệm trên LLM có “long context” (đọc offline toàn bộ)

  

- Thử nghiệm GPT-4o, Llama 3.1 (8B & 70B) cùng mô hình Phi-3 Medium (128k context) trên **LONGMEMEVAL-S** (~115k tokens).
    
- Kết quả: khi so với “oracle” (chỉ đưa đúng những đoạn cần thiết), tỉ lệ trả lời đúng giảm mạnh (30%–60%).
    
- Cho thấy: kể cả LLM với ngữ cảnh dài thì vẫn cần cơ chế memory & retrieval tốt để xử lý lịch sử quá lớn.
    

  

---

  

## 4. Thiết kế giải pháp Memory-Augmented Chatbot

  

Bài báo đưa ra một **khung tổng quát** gồm 3 giai đoạn và 4 “điểm điều khiển”:

  

1. **Indexing**: Lưu trữ mỗi phiên (hoặc chia nhỏ hơn) dưới dạng cặp (key, value).
    
2. **Retrieval**: Tìm kiếm (top-k) các key phù hợp với query.
    
3. **Reading**: Mô hình LLM đọc và suy luận trên các đoạn (value) truy xuất được.
    

  

**4 điểm điều khiển**:

  

- **CP1 (Value)**: Lựa chọn mức độ chia nhỏ hoặc tóm tắt session (theo round/phiên hay trích xuất facts).
    
- **CP2 (Key)**: Có thể “mở rộng key” bằng cách gắn thêm summary, keyphrase, facts quan trọng… giúp tăng khả năng tìm kiếm.
    
- **CP3 (Query)**: Đặc biệt chú ý đến **temporal query** (dùng mốc thời gian) để lọc bớt dữ liệu cũ hoặc không liên quan.
    
- **CP4 (Reading Strategy)**: Áp dụng dạng prompt “Chain-of-Note” (CoN) hoặc định dạng cấu trúc (JSON) để mô hình trích rút thông tin trước rồi mới kết luận. Phòng ngừa lỗi khi LLM đọc nhiều đoạn dài.
    

  

### Một số kết quả thực nghiệm quan trọng

  

- **(CP1) Decomposition**: Thay vì lưu cả session, chia nhỏ theo “round” (mỗi lượt người dùng) giúp tăng khả năng tìm kiếm. Tuy nhiên, tóm tắt/facts quá nhiều cũng có rủi ro “mất thông tin”.
    
- **(CP2) Key expansion**: Thêm “facts, keyphrase, summary” vào key có thể tăng Recall@k và NDCG@k, dẫn đến tăng 4–5% độ chính xác cuối cùng.
    
- **(CP3) Thêm dữ liệu thời gian**: Nếu câu hỏi liên quan thời gian, trích xuất khung thời gian (time range) và lọc value giúp độ chính xác truy xuất tăng 7–11%. Model mạnh như GPT-4o “hiểu” thời gian tốt hơn model nhỏ.
    
- **(CP4) Reading**: Dù truy xuất đúng, LLM vẫn có thể sai do bối cảnh dài. Dùng “Chain-of-Note” + format JSON làm mô hình phân tách việc trích rút thông tin và lập luận. Cải thiện 5–10% so với prompt thông thường.
    

  

---

  

## 5. Kết luận

  

Bài báo giới thiệu **LONGMEMEVAL** – một bộ dữ liệu kiểm tra tổng quát nhiều khía cạnh của bộ nhớ dài hạn cho hệ thống hội thoại AI. Tác giả chỉ ra:

  

- Dữ liệu có quy mô rất lớn (trên 100k tokens / trường hợp), đa dạng loại câu hỏi (7 kiểu), bao quát 5 năng lực quan trọng (IE, MR, TR, KU, ABS).
    
- Qua thử nghiệm, các mô hình và dịch vụ hàng đầu vẫn gặp khó khi phải duy trì “state” lâu dài; do đó **bắt buộc cần thiết** tích hợp các kỹ thuật indexing–retrieval–reading tinh vi.
    
- Bài báo đề xuất một “khung thiết kế” (key, value, query, reading strategy) và chứng minh các kỹ thuật như chia nhỏ session, key expansion, xử lý time-aware, và “Chain-of-Note” đều giúp nâng cao độ chính xác.
    

  

**LONGMEMEVAL** là bước tiến lớn trong việc đánh giá khả năng “ghi nhớ dài hạn” của trợ lý hội thoại, tạo tiền đề cho các nghiên cứu sâu hơn về tối ưu hóa chi phí, thời gian, hoặc mở rộng sang nhiều dạng dữ liệu phức tạp hơn.

  

  

---

2. # Đào sâu hơn
    

## 2.1 Câu hỏi 1: Đào sâu hơn cách làm

  

Bài báo định nghĩa **Value** là đoạn hội thoại hoặc lượt tương tác (session hoặc round), và Key là cách biểu diễn value để lập chỉ mục (index).

|   |   |
|---|---|
|Key type|Cách tạo|
|summary|Dùng LLM để tóm tắt hội thoại (zero-shot prompt, xem Fig.11)|
|fact|Trích xuất thông tin cá nhân, sự kiện, thực thể... từ đoạn hội thoại (prompt dạng JSON)|
|keyphrase|Trích cụm từ khóa chính (dùng prompt dạng liệt kê, không cần fine-tune)|
|V + …|Ghép đoạn hội thoại gốc (V) với summary / fact / keyphrase để tạo key mở rộng|

→ Chi tiết cách làm có trong phần **Appendix E – Memory Optimizations: Implementation Details** và **Figure 11**

  

---

  

![[Pasted image 20250322054315.png]]

  

  

Hình vẽ trong bài mô tả **quy trình “3 giai đoạn” (Indexing – Retrieval – Reading)** của một hệ thống “trợ lý hội thoại có bộ nhớ dài hạn,” đồng thời chỉ ra **4 điểm điều khiển (CP1–CP4)** quan trọng:

  

## 2.1 Pipeline `Indexing – Retrieval – Reading` kèm theo 4 **Control Points (CP1–CP4)**, mỗi bước gồm: chức năng, thành phần chính, ví dụ cụ thể và cách tối ưu:

---

### 📊 **Chi tiết Pipeline: Indexing – Retrieval – Reading với CP1–CP4**

|   |   |   |   |   |   |
|---|---|---|---|---|---|
|Giai đoạn|Control Point|Vai trò chính|Thành phần chính|Ví dụ minh họa|Cách tối ưu hóa|
|1. Indexing|CP1: Key|Biểu diễn thông tin để phục vụ tìm kiếm truy vấn sau này|Summary / Keyphrase / Fact / Embedding|Key = V + fact → đoạn hội thoại + "User likes science fiction"|- Dùng LLM để sinh key giàu ngữ nghĩa  <br>- Kết hợp V + fact để tăng tính phủ khớp|
||CP2: Value|Nội dung lưu trữ thực tế, là đầu vào cho mô hình đọc|Round / Session / Document Chunk|Một phiên tư vấn 5 lượt trò chuyện về sở thích phim|- Chọn granular phù hợp (round/session)  <br>- Tách câu thừa, tối ưu nội dung cần đọc|
|2. Retrieval|CP3: Query|Truy vấn từ người dùng → dùng để so khớp với Key trong chỉ mục|User query + Query Expansion + Time tags|"What kind of movies did I say I like last week?"|- Làm giàu truy vấn bằng query expansion  <br>- Thêm context: thời gian, entity, intent|
||||Dense retriever, reranker, time-aware|Dùng retriever để tìm ra Key gần nhất → lấy top-k Value tương ứng|- Sử dụng retriever mạnh (Stella, Contriever, etc.)  <br>- Áp dụng session-aware hoặc hybrid filtering|
|3. Reading|CP4: Reading Strategy|Đưa Value vào LLM cùng query → sinh ra câu trả lời cuối cùng|Prompt template, Chain-of-Note, JSON format|Đưa 3 đoạn Value vào LLM: "Based on the following memories, answer the question…"|- Dùng định dạng prompt rõ ràng (JSON, note)  <br>- Chia 2 bước: rút ý – lý luận (Chain-of-Note)|

---

### 🧠 Ghi chú:

- **CP1–CP4** là những điểm can thiệp (Control Points) giúp tối ưu chất lượng truy hồi và độ chính xác cuối cùng.
    
- Đây là **cách tổ chức truy hồi có cấu trúc (structured retrieval)**, rất phù hợp với hệ thống hỏi đáp có **trí nhớ dài hạn (long-term QA)**.
    
- `K = V + fact` và `Chain-of-Note` là hai kỹ thuật được chứng minh là hiệu quả nhất trong bài **LONGMEMEVAL**.
    

  

  

---

  

## Bối cảnh giả định

  

- Bạn có một trợ lý AI (dựa trên LLM) hỗ trợ theo dõi quá trình tập luyện thể thao của bạn trong nhiều tuần lễ.
    
- Bạn thường xuyên trò chuyện với trợ lý, cung cấp thông tin số km chạy bộ, thời gian nghỉ, cân nặng, v.v.
    

  

### Lịch sử hội thoại

  

1. **Tuần 1**:
    
    2. **User**: “Tuần này tôi chạy 15km. Tôi muốn tăng dần cường độ.”
        
    3. **Assistant**: “Ok, vậy tuần sau hãy thử nâng lên 20km.”
        
2. **Tuần 2**:
    
    2. **User**: “Tôi đã chạy được 18km. Nhưng có vẻ hơi mệt.”
        
    3. **Assistant**: “Có thể bạn cần nghỉ xen kẽ 1 ngày sau mỗi 2 buổi chạy.”
        
3. **Tuần 3**:
    
    2. **User**: “Tôi chạy 21km trong tuần này. Rất tốt!”
        
    3. **Assistant**: “Chúc mừng! Hãy ghi nhật ký lại để xem tuần tiếp theo thế nào.”
        

  

… (và còn tiếp)

  

---

  

## 1) Indexing

  

### CP1: Key

  

- Mỗi phiên/đoạn hội thoại sẽ được “ghi chỉ mục” (index) sao cho khi cần tìm lại thông tin, hệ thống có thể dựa vào **key**.
    
- **Ví dụ**: Ta lấy **câu User** làm “Value” chính cần lưu, còn **key** có thể là:
    
    - Embedding (vector) được sinh ra từ toàn bộ nội dung tin nhắn,
        
    - hoặc một cụm từ tóm tắt như “User chạy 15km tuần 1” (v.v.).
        

  

**Giả sử** với Week 1, ta có:

  

- **Key** = `"tuần 1, chạy bộ, 15km, cường độ"` (có thể kết hợp vector embedding + từ khóa)
    
- **Value** = `"Tuần này tôi chạy 15km. Tôi muốn tăng dần cường độ."`
    

  

### CP2: Value

  

- “Value” là **nội dung gốc** dùng để phục vụ trả lời. Ở đây, mỗi lượt nói của User là một “value chunk.”
    
- Có thể chia nhỏ hơn nữa (mỗi câu 1 chunk) hoặc giữ nguyên cả phiên tùy bạn thiết kế.
    
- Nếu muốn **tóm gọn** (summarize) hoặc trích xuất “fact” để tiết kiệm dung lượng, ta vẫn phải đảm bảo không làm mất thông tin cần thiết.
    

  

**Ví dụ**:

  

- Tuần 2 → **Value** = “Tôi đã chạy được 18km. Nhưng có vẻ hơi mệt.”
    
- Key có thể mở rộng: `"tuần 2, chạy bộ, 18km, mệt mỏi"`
    

  

---

  

## 2) Retrieval

  

### CP3: Query

  

- Khi bạn hỏi: “Tôi đã chạy được bao nhiêu km mỗi tuần, và tuần nào tôi chạy nhiều nhất?”
    
- Hệ thống sẽ tạo một **truy vấn** (query) dựa trên chính câu hỏi này.
    
- Trong **temporal query** (liên quan thời gian), nếu model phát hiện từ khóa “mỗi tuần” và “tuần nào chạy nhiều nhất,” nó có thể chỉ tìm các **key** chứa cụm “chạy bộ” + “km” + “tuần #”.
    

  

**Quá trình tìm kiếm**:

  

1. So sánh **query embedding** với các **key** trong cơ sở dữ liệu.
    
2. Lấy top-k Value chunk liên quan, ví dụ:
    
    1. Week 1 (15km), Week 2 (18km), Week 3 (21km), …
        

  

(Chúng ta có thể có thêm bước lọc thời gian, ví dụ: chỉ lấy các tuần <= “tuần hiện tại”)

  

---

  

## 3) Reading

  

### CP4: Reading Strategy

  

- Bây giờ, hệ thống có được 3 “value chunk” (tuần 1, 2, 3). Nó đưa những chunk này + câu hỏi vào LLM để tổng hợp ra kết quả.
    

- **Ví dụ** “Chain-of-Note” (CoN):
    
    - **Bước 1**: Mô hình **trích rút** các con số liên quan từ mỗi chunk:
        
        1. Tuần 1: 15km, tuần 2: 18km, tuần 3: 21km.
            
    - **Bước 2**: Mô hình **suy luận**: “Tuần 3 có quãng đường dài nhất (21km).”
        
    - **Bước 3**: Trả lời người dùng ngắn gọn: “Bạn đã chạy 15km, 18km, 21km… Nhiều nhất là tuần 3.”
        
- Nếu **không** dùng kỹ thuật này, mô hình có thể “lẫn” thông tin hoặc bỏ sót dữ liệu, đặc biệt khi có nhiều chunk.
    

  

---

  

## Tóm tắt:

  

- **Indexing** (1) & (2):
    
    - **CP1** (Key): quyết định dạng khóa (embedding, từ khóa, tóm tắt…).
        
    - **CP2** (Value): quyết định mức “chia nhỏ” dữ liệu (theo vòng hội thoại, theo câu, hay tóm tắt).
        
- **Retrieval** (3):
    
    - **CP3** (Query): mở rộng (query expansion), hiểu thời gian, từ khóa… để tìm chính xác.
        
- **Reading** (4):
    
    - **CP4** (Reading Strategy): chuỗi prompt hoặc kỹ thuật (Chain-of-Note, JSON format, v.v.) để LLM diễn giải tốt nhiều đoạn.
        

  

Qua ví dụ này, ta thấy rõ cách “Key–Value” được lưu và cách “Query–Reading Strategy” can thiệp để tối ưu việc tìm kiếm và tổng hợp thông tin.

  

---

## 2.2 Câu hỏi thêm 2: Theo tôi hiểu là bạn sử dụng chiến lược Indexing Key - Value trong đó key = embedidng + từ khóa Value = toàn bộ text lúc search là hybrid search đúng chứ

  

**Đúng như bạn nói,** trong bài (và cả nhiều hệ thống RAG tương tự), một **cách phổ biến** là lưu trữ “Key” dưới dạng **kết hợp (embedding + từ khoá)**, rồi **Value** là toàn bộ text.

  

- **Key** thường gồm:
    
    - **Embedding** (vector) để phục vụ “semantic search” (tìm kiếm theo ngữ nghĩa).
        
    - **Từ khoá** (hoặc tóm tắt, fact...) để phục vụ “keyword / BM25 search” (hoặc hỗn hợp).
        

  

Khi hệ thống **tìm kiếm (retrieval)**, nó có thể chạy cả hai cơ chế:

  

- Tính “độ tương đồng embedding” (cosine similarity, v.v.)
    
- Tìm theo từ khoá (BM25 hay từ vựng đặc biệt) Sau đó ta có thể **gộp** hai danh sách kết quả (thuật toán rank fusion) hoặc chọn lọc top-k phù hợp. Đó chính là **“hybrid search”**.
    

  

Tùy nhu cầu, **cách triển khai** có thể là:

  

1. **“Key merging”**: Trộn thẳng từ khóa/tóm tắt vào text để tạo **một “key” duy nhất** (có embedding + fulltext).
    
2. **“Rank merging”**: Lưu song song 2 loại key (embedding key + keyword key), khi cần truy vấn thì trả về 2 list kết quả rồi hợp nhất.
    

  

Cả hai chiến lược đều nhằm tăng khả năng tìm được văn bản chính xác (recall) và ưu tiên mức độ liên quan (precision). Nhiều bài thực nghiệm cho thấy cách phối hợp embedding + keyword thường hiệu quả hơn so với chỉ dùng đơn lẻ một trong hai.

  

---

## 2.3 Câu hỏi 3: Trong bài báo có nói dùng cách nào để extract từ khóa không ?

  

Trong bài báo, phần **Appendix E** (Implementation Details) có mô tả cách tác giả **extract keyphrase** từ mỗi “chunk” (một phiên hội thoại hoặc một lượt nói). Họ dùng một **prompt zero-shot** với mô hình _Llama 3.1 8B Instruct_, có nội dung như sau (được rút gọn):

  

```Plain
Below is a transcript of a conversation between a human user and an AI assistant. 
Generate a list of keyphrases for the session. Separate each keyphrase with a semicolon. 
Dialogue content: {session}
```

  

Cụ thể:

  

1. **Mỗi phiên (hoặc lượt) hội thoại** được đưa vào prompt dưới dạng `Dialogue content`.
    
2. **Mô hình** Llama 3.1 8B Instruct **trích xuất ra** các **từ khóa nổi bật**, dưới dạng danh sách, ngăn cách bằng dấu chấm phẩy.
    
3. Những keyphrases này sau đó được nối (concatenate) vào Value hoặc ghép chung dưới dạng “document expansion” để làm Key.
    

  

Tóm lại, tác giả không dùng thuật toán keyword extraction truyền thống (như TF-IDF, TextRank, v.v.) mà **chỉ dùng prompt zero-shot** với mô hình ngôn ngữ lớn để trích xuất từ khóa. Kết quả thu được được họ báo cáo trong bảng ở phần phụ lục.

  

  

## 2.4 Tức là trích xuất sau khi hội thoại đã kết thúc hả ?

  

```Plain
Trong bài, họ chạy **quy trình “extract keyphrase”** cho từng khối hội thoại (chunk) **sau khi** khối đó đã sẵn sàng (tức là sau khi người dùng và trợ lý đã kết thúc phiên hoặc ít nhất là kết thúc một lượt nói). Ta có thể coi đây là quá trình “post-hoc extraction”: khi **đoạn text** của một lượt/phiên đã hoàn chỉnh, họ đưa đoạn đó vào prompt để LLM trích xuất từ khóa.

- Về mặt triển khai, bài báo chủ yếu mô tả **quy trình** dưới góc nhìn “offline benchmark”: họ có toàn bộ dữ liệu hội thoại, sau đó chạy lần lượt các bước Indexing – Retrieval – Reading.
- Nếu áp dụng vào hệ thống “online memory”, ta cũng thường **chờ** người dùng nói xong (hay trợ lý trả lời xong) rồi mới trích xuất từ khóa/facts để cập nhật vào bộ nhớ.

Tóm lại, **không** phải họ trích xuất từng câu “ngay lập tức khi người dùng gõ”, mà **đợi** cho đoạn/lượt thoại hoàn tất, sau đó mới đưa vào LLM để sinh keyphrase.
```

  

  

![[Pasted image 20250322060614.png]]

  

Dưới đây là phần giải thích cụ thể cách đọc **Bảng 2** (Retrieval và End-to-End QA) trong bài báo, cũng như ý nghĩa từng cột:

  

---

  

## 1. Các cột về **Retrieval**

  

- **Metrics@5** và **Metrics@10**: Tác giả báo cáo hai chỉ số:
    
    - **Recall**@k: Tỉ lệ mà top-k kết quả trả về (từ bước tìm kiếm) có chứa **đoạn/bản ghi** thực sự cần để trả lời (các “evidence” quan trọng).
        
    - **NDCG**@k: Thước đo đánh giá mức độ xếp hạng (nếu “evidence” càng được xếp hạng cao, thì NDCG càng tốt).
        
- Bên trái cùng bảng là phần “Key Design” (cách tạo Key trong Indexing) và “Value = Round/Session” (cách chia “Value chunk”).
    

  

**Ví dụ**:

  

- Khi thấy “K = V + fact” có Recall@5 = 0.644 (với **Value = Round**), có nghĩa là nếu ta **key = [Value + facts rút ra]**, thì khi truy vấn, có 64.4% trường hợp hệ thống tìm thấy đúng “chunk” liên quan trong top 5 kết quả.
    

  

---

  

## 2. Các cột về **End-to-End QA**

  

Sau khi truy xuất (Retrieval), các top-k “chunk” được đưa vào LLM để trả lời. Bảng hiển thị **độ chính xác cuối** (QA accuracy) theo 2 mức k:

  

- **Top-5**: Chỉ lấy 5 đoạn được xếp hạng cao nhất đưa vào LLM
    
- **Top-10**: Lấy 10 đoạn. Mục đích là so sánh xem đưa thêm chunk (top-10) có cải thiện câu trả lời so với chỉ top-5 hay không.
    

  

Ba cột “GPT-4o,” “L3.1 70B,” “L3.1 8B” chính là ba “độc giả” (reader LLM) khác nhau. Mỗi LLM sẽ có khả năng tổng hợp thông tin từ top-k chunk tốt hay kém, dẫn đến kết quả QA khác nhau.

  

---

  

## 3. Ý nghĩa từng “Key Design”

  

- **K = V**: Key trùng với Value. Tức là không tách bạch, dùng luôn toàn bộ nội dung text để làm khóa tìm kiếm.
    
- **K = summary, K = fact, K = keyphrase**: Key là phần tóm tắt, facts hay keywords rút gọn (không kèm đoạn gốc).
    
- **K = V + summary / V + fact / V + keyphrase**: Kết hợp cả đoạn gốc (Value) và phần tóm tắt/fact/keyword rút ra để mở rộng nội dung Key (document expansion).
    

  

---

  

## 4. Các hàng “Value = Round” và “Value = Session”

  

- **Value = Round**: Mỗi lượt nói của người dùng (hoặc mỗi câu) được cắt thành một “chunk” riêng.
    
- **Value = Session**: Dùng cả phiên hội thoại (gồm nhiều lượt nói) làm một chunk “Value”.
    

  

Cột “Value = Round” thường có nhiều chunk hơn (nhưng ngắn), còn “Value = Session” có số chunk ít hơn (nhưng dài).

  

---

  

## 5. Rút gọn các nhận xét

  

1. **Retrieval**:
    
    2. Thường thấy “K = V + fact” (bôi đậm trong bảng) có Recall và NDCG cao hơn đáng kể so với K = V đơn thuần. Điều này chứng tỏ việc “document expansion” với facts (hoặc tóm tắt) làm Key giúp việc tìm kiếm chính xác hơn.
        
2. **End-to-End QA**:
    
    3. Tương tự, “K = V + fact” có accuracy cao nhất ở hầu hết các cột (so với K = summary/keyphrase riêng lẻ).
        
    4. Việc so sánh “Top-5” và “Top-10” giúp ta thấy mô hình GPT-4o, Llama-3.1 70B, 8B thay đổi hiệu quả ra sao khi bơm thêm chunk. Ví dụ GPT-4o thường tăng từ ~0.65 lên ~0.70 nếu lấy Top-10 thay vì Top-5.
        
3. **Value = Round** vs. **Value = Session**:
    
    2. Nhìn chung, “Value = Round” có recall thấp hơn do độ ngắn (có nhiều chunk nhỏ, khó match), nhưng đôi khi “Value = Round” hỗ trợ mô hình QA trả lời chính xác hơn vì chunk gọn, mô hình ít bị loãng thông tin.
        
    3. “Value = Session” thì retrieval recall có thể cao hơn (vì chunk to, dễ dính key), nhưng khi vào QA, có thể mô hình bị “khó tóm gọn” nên kết quả cuối có thể khác nhau.
        

  

---

  

### Kết luận chính từ bảng

  

- **Document expansion** (K = V + fact) **cải thiện cả retrieval** (Recall/NDCG) **lẫn kết quả QA**.
    
- Nếu chia **Value** thành từng **round** nhỏ thì có thể tốt cho giai đoạn tổng hợp, nhưng đòi hỏi key design cẩn thận (vì chunk nhỏ, retrieval cũng khó).
    
- Việc chọn top-5 hay top-10 chunk khi đưa vào LLM cũng ảnh hưởng đáng kể đến hiệu suất cuối, và mô hình LLM mạnh như GPT-4o thường tận dụng tốt hơn top-10 so với model nhỏ.
    

  

Như vậy, Bảng 2 minh họa chi tiết **các tổ hợp “Key Design” (K) và “Value Granularity”** sẽ ảnh hưởng thế nào đến cả hiệu năng truy xuất (retrieval) lẫn chất lượng trả lời cuối (end-to-end QA).

  

  

**Đúng vậy**, trong phương án **Key = V + fact** thì ta tạo **một chuỗi “Key”** bằng cách **ghép** (concatenate) phần **nội dung gốc (Value)** với các câu “fact” rút ra, rồi **mang cả chuỗi đó đi nhúng (embedding)** và lưu vào cơ sở dữ liệu để truy xuất.

  

Nói cách khác:

  

1. **Value**: nội dung gốc của đoạn (có thể là nguyên phiên hoặc chỉ một lượt).
    
2. **Fact**: các câu thông tin cốt lõi trích xuất (ví dụ, tóm tắt “người dùng vừa mua xe, tuần này chạy 15km,…”).
    
3. **Ghép (Value + fact)** thành một đoạn text duy nhất, rồi đưa text này qua mô hình embedding (VD: BERT, Stella V5, Contriever...).
    
4. **Lưu embedding** đó làm “Key” trong chỉ mục. Khi truy vấn, ta sẽ so sánh query embedding với embedding của “Key = (Value + fact)”.
    

  

Cách làm này giúp hệ thống tìm kiếm “match” tốt hơn, vì từ khoá cốt lõi (fact) cũng nằm trong chính “Key”, đồng thời vẫn giữ bối cảnh gốc (Value). Ta có thể coi đó là **một dạng “document expansion”** – thêm thông tin tóm lược, facts, keyphrase… vào văn bản gốc trước khi nhúng để tăng độ chính xác truy xuất.

  

---

## 2.6 Phân biệt fact với 2 cái summary và keyphrase?

- **Summary**: viết thành **đoạn văn mô tả** tổng quan, có độ bao quát cao.
    
- **Fact**: chia nhỏ thành **nhiều câu đơn** (mỗi câu là một sự thật riêng).
    
- **Keyphrase**: chỉ là **cụm từ khóa** chứ không viết thành câu hoàn chỉnh.
    

![[Pasted image 20250322063052.png]]

  

## Sự khác nhau giữa fact và summary

**fact** (các dữ kiện rời) và **summary** (một đoạn tóm tắt) khi trích xuất nội dung:

  

> **User**: “Tôi vừa chuyển từ iPhone 13 sang điện thoại Android, cụ thể là **Samsung Galaxy S23**. Tôi dùng iPhone 13 khoảng **1 năm** rồi, giờ đổi qua S23 thấy **camera đẹp** hơn hẳn.”

  

---

  

### **1) Facts**

  

Facts thường được biểu diễn dưới dạng **các câu thông tin rời rạc, ngắn gọn**, mỗi fact nêu một điểm quan trọng:

  

1. “Người dùng **đã dùng iPhone 13** trong 1 năm.”
    
2. “Người dùng **chuyển sang sử dụng Samsung Galaxy S23** (Android).”
    
3. “Người dùng **thấy camera Samsung S23 đẹp hơn**.”
    

  

**Đặc trưng**:

- Từng dòng fact là một **thực thể quan hệ** (VD: “User – chuyển sang – Galaxy S23”).
    
- Không có câu nối; tránh diễn giải, cảm xúc.
    
- Lượng thông tin tối thiểu, dễ “match” khi tìm kiếm.
    

  

---

  

### **2) Summary**

  

Phần tóm tắt (summary) thường là **một đoạn văn** mô tả chung những điểm chính trong hội thoại:

  

> “Người dùng cho biết họ đã **sử dụng iPhone 13** khoảng 1 năm trước khi **chuyển sang điện thoại Android**. Họ hiện dùng **Samsung Galaxy S23** và cảm thấy hài lòng, đặc biệt với **chất lượng camera** được đánh giá cao hơn so với iPhone 13.”

  

**Đặc trưng**:

  

- Viết thành **đoạn văn liên tục**, tương đối mạch lạc.
    
- Có thể lồng ghép các **mối quan hệ**, bối cảnh, và cả đánh giá (cảm xúc).
    
- Thích hợp cho việc **tóm gọn** nhiều ý, giúp mô hình đọc dễ hơn, nhưng có thể **lược bớt chi tiết**.
    

  

## 2.7 **Tóm tắt cách tạo Key (summary, fact, keyphrase)**

Trong bài báo **LONGMEMEVAL**, tác giả mô tả chi tiết cách tạo các loại **Key** (K) để đại diện cho mỗi đoạn hội thoại (Value – V) trong hệ thống memory indexing:

|   |   |   |
|---|---|---|
|Loại Key|Cách tạo chi tiết|Công cụ dùng|
|summary|Tóm tắt toàn bộ đoạn hội thoại một cách ngắn gọn, tập trung vào những thông tin chính do người dùng nêu ra.|Prompt + LLaMA 3.1 8B|
|fact|Trích xuất các sự thật cá nhân hóa từ lời nói của người dùng như sở thích, hành vi, sự kiện, số liệu... dạng câu đơn độc lập.|Prompt + LLaMA 3.1 8B (dùng few-shot)|
|keyphrase|Trích xuất các cụm từ khóa chính, cách ngắn gọn hơn summary.|Prompt + LLaMA 3.1 8B|

✅ Sau khi tạo được các summary/fact/keyphrase, **các tác giả mở rộng key bằng cách ghép nối** chúng với Value gốc để tạo thành: → `K = V + summary`, `K = V + fact`, `K = V + keyphrase`.

---

  

```Plain
1. Thay vì chỉ có: K = V + fact/V+ summary/V+keyphrase thì ta dùng cả K = V+fact+summary +keyphrase và sử dụng các method tốt hơn nữa để summary ???

**Áp dụng “hierarchical indexing”**:

- Thay vì key = “(V + fact + summary + keyphrase)” dạng phẳng, ta tách thành 2 tầng:
    - Tầng 1: Từ khoá/summaries tổng quát (để khoanh vùng).
    - Tầng 2: Chuẩn hoá fact/câu gốc (để lấy chính xác).
- Đọc RAG theo hai pha, giảm thời gian “cày” trên hàng chục nghìn “chunk.”

2. fine tuning model, sử dụng các history để fine tuning model làm tăng khả năng extract, 1 model LMa 3 - 3B thay vì llama3.1 8B của tác giả ...

3. ...
```

  

Giống cái a Huy

```Plain
- Raptor Chunk 
- LLMs Chunk 
-> a Huy kết hợp cả 2 
```

  

## 2.8 Đọc kỹ lại K = value + fact của họ

  

Câu hỏi của bạn là về phần:

  

> **"K = V + fact" – quy trình này được thực hiện như nào trong toàn bộ pipeline? Có phải là V + fact được mang đi embedding không?**

  

**Có, đúng là** **`V + fact`**** được mang đi embedding** trong quá trình indexing — đây là kỹ thuật gọi là **Key Expansion (mở rộng khóa)** bằng cách **nối (**`**concat**`**) thông tin quan trọng như facts vào chính **`**value**` và dùng toàn bộ chuỗi đó như **key** trong cặp key-value. Điều này giúp tăng khả năng truy xuất đúng (retrieval) và cải thiện độ chính xác của trả lời câu hỏi.

### 1. **Giải thích các khái niệm**

  

- `V`: Giá trị lưu trong memory (ví dụ: một đoạn hội thoại, một vòng chat, hoặc bản tóm tắt).
    
- `fact`: Những thông tin cá nhân được trích xuất từ `V` như sở thích, trải nghiệm, con số cụ thể, v.v.
    
- `K = V + fact`: Là **Key mở rộng**, được tạo ra bằng cách nối `fact` vào `value` rồi mang toàn bộ đi **embedding** để tạo thành key.
    

  

---

  

### 2. **Trong pipeline của hệ thống memory-augmented chat assistant**

Theo mô hình được mô tả trong Figure 3 (trang 4 của bài), pipeline có 3 bước chính:

#### **(1) Indexing:**

- Mỗi session được chia nhỏ (tốt nhất là theo từng vòng `round`).
    
- Từ mỗi vòng, ta tạo ra:
    
    - `value` (V): thường là đoạn hội thoại người dùng.
        
    - `fact`: trích xuất các thông tin có cấu trúc (user facts).
        
- **Sau đó,** `**key = V + fact**`: nối lại rồi nhúng embedding → lưu vào bộ nhớ key-value.
    

📌 **Việc này giúp "gợi ý" cho retriever biết cái gì quan trọng trong **`**V**`**.**

---

#### **(2) Retrieval:**

- Khi người dùng hỏi một câu (`query`), nó được nhúng thành embedding.
    
- Hệ thống tìm top-k `key` gần nhất trong vector space, lấy `value` tương ứng.
    

> 🧠 Việc mở rộng `K = V + fact` giúp tăng Recall@k và QA accuracy như được chứng minh trong Table 6 và Table 7.

---

#### **(3) Reading:**

- Các `value` được truy xuất sẽ được model đọc lại để suy luận và trả lời.
    
- Thường kết hợp với kỹ thuật như **Chain-of-Note** hoặc **JSON format** để tăng độ chính xác.
    

---

## 🧪 Kết quả thực nghiệm

  

Trong các bảng kết quả (như Table 6 trang 17):

- `K = V + fact` vượt trội hơn `K = V` hoặc `K = fact` về cả **retrieval** và **QA accuracy**.
    
- Đặc biệt hiệu quả khi dùng với value granularity = session hoặc round.
    

  

---

## 2.9 **Giải thích học thuật vì sao K = V + fact hiệu quả nhất**

#### 🔍 (a) **Fact có tính chọn lọc cao (high selectivity)**

- Các _fact_ là các thông tin cô đọng, trọng yếu, giúp **giảm nhiễu** khi lập chỉ mục.
    
- Điều này giúp **retriever** dễ tìm được thông tin khớp truy vấn hơn.
    

> “We find applying document expansion with the extracted user facts… greatly improves both retrieval and downstream QA.” — Table 6Read2 - LONGMEMEVAL - B…

#### 🧩 (b) **Ghép V + fact cân bằng chi tiết & trừu tượng**

- V chứa **ngữ cảnh gốc**, còn fact chứa **tóm tắt ngữ nghĩa**.
    
- Điều này giúp hệ thống phản hồi được **cả truy vấn chi tiết lẫn truy vấn khái quát**.
    

> “This approach… yielded an average improvement of 4% in retrieval metrics and 5% in final accuracy.” — Appendix D.4Read2 - LONGMEMEVAL - B…

#### 💡 (c) **Semantic richness giúp match sâu**

- _Fact_ thường chứa **entity**, **attribute**, **action** → rất khớp với loại câu hỏi trong QA như:
    
    - “When did I say I liked science fiction?”
        
    - “What did I mention about my parents?”
        

→ Các thông tin này rất phù hợp với **dense retrievers** và LLM-based readers.

#### 🔁 (d) **Tương thích với retriever hiện đại**

- Dense retrievers như **Stella V5**, **Contriever** hoạt động tốt hơn khi index chứa thông tin trích xuất (fact) thay vì đoạn gốc dài.
    
- Việc ghép `V + fact` giúp tăng **semantic overlap** → tăng **recall** và **accuracy** trong downstream QA.
    

> “Combining value and fact into keys significantly improves retrieval and QA results across all retrievers.” — Table 6, Table 9Read2 - LONGMEMEVAL - B…

---

### 🧠 Tổng kết lý do K = V + fact vượt trội:

|   |   |
|---|---|
|Lý do|Ảnh hưởng tích cực|
|Fact có tính khái quát cao|Giảm nhiễu, tăng độ chính xác khi truy hồi|
|Ghép V + fact = đa dạng ngữ cảnh|Bao phủ tốt cả chi tiết và tóm lược → dễ match hơn|
|Semantic richness từ fact|Khớp với câu hỏi phức tạp, lý luận → tăng QA accuracy|
|Tối ưu cho retriever hiện đại|Phù hợp với dense retriever và LLM-based readers|

## 2.10 Prompting

Chào Quốc,

Dưới đây là **giải thích ngắn gọn và dễ hiểu** về các loại **prompt** được sử dụng trong bài báo **LongMemEval**:

---

### 1. **Tóm tắt (Summaries)**

**Mục tiêu**: Tạo một đoạn tóm tắt ngắn gọn về nội dung chính của cuộc hội thoại giữa người dùng và trợ lý AI.

**Cách thực hiện**: Sử dụng mô hình ngôn ngữ lớn (LLM) để tạo một đoạn văn ngắn, tập trung vào những chủ đề chính và thông tin quan trọng mà người dùng đã đề cập hoặc yêu cầu.

**Ví dụ Prompt**:

```Plain
Dưới đây là bản ghi cuộc trò chuyện giữa một người dùng và một trợ lý AI. Vui lòng tóm tắt đoạn hội thoại này một cách ngắn gọn nhất có thể trong một đoạn văn ngắn, trích xuất các chủ đề chính và thông tin quan trọng. Trong bản tóm tắt của bạn, tập trung nhiều hơn vào những gì người dùng đã đề cập hoặc yêu cầu.
Nội dung hội thoại: {session or round}
```



---

### 2. **Từ khóa chính (Keyphrases)**

**Mục tiêu**: Trích xuất các cụm từ khóa quan trọng từ cuộc hội thoại để dễ dàng tìm kiếm và truy xuất sau này.

**Cách thực hiện**: Sử dụng LLM để tạo danh sách các cụm từ khóa đại diện cho nội dung chính của cuộc hội thoại, mỗi cụm từ được phân tách bằng dấu chấm phẩy.

**Ví dụ Prompt**:

```Plain
Dưới đây là bản ghi cuộc trò chuyện giữa một người dùng và một trợ lý AI. Hãy tạo một danh sách các cụm từ khóa cho phiên này. Mỗi cụm từ khóa được phân tách bằng dấu chấm phẩy.
Nội dung hội thoại: {session or round}
```



---

### 3. **Thông tin cá nhân của người dùng (User Facts)**

**Mục tiêu**: Trích xuất các thông tin cá nhân, sự kiện trong đời, kinh nghiệm và sở thích liên quan đến người dùng từ các tin nhắn của họ.

**Cách thực hiện**: Sử dụng LLM để xác định và liệt kê các thông tin cá nhân dưới dạng danh sách JSON, mỗi phần tử là một câu đơn giản mô tả một sự kiện hoặc sở thích cụ thể của người dùng.

**Ví dụ Prompt**:

```Plain
Bạn sẽ nhận được một danh sách các tin nhắn từ một người dùng gửi đến một trợ lý AI. Hãy trích xuất tất cả thông tin cá nhân, sự kiện trong đời, kinh nghiệm và sở thích liên quan đến người dùng. Đảm bảo bao gồm tất cả các chi tiết như sự kiện trong đời, kinh nghiệm cá nhân, sở thích, số liệu cụ thể, địa điểm hoặc ngày tháng. Nêu mỗi thông tin trong một câu đơn giản. Đặt những câu này trong một danh sách JSON, mỗi phần tử là một sự kiện cá nhân độc lập về người dùng. Giảm thiểu việc sử dụng đại từ bằng cách thay thế bằng các thực thể cụ thể. Nếu không có sự kiện cụ thể, thông tin cá nhân hoặc sở thích nào được đề cập, chỉ cần tạo một danh sách trống.
Tin nhắn của người dùng: {session}
Thông tin cá nhân về người dùng (một danh sách các chuỗi ở định dạng JSON; không tạo thêm bất kỳ nội dung nào khác):
```



---

**Lưu ý**: Các prompt trên được thiết kế để hướng dẫn mô hình ngôn ngữ trích xuất thông tin một cách hiệu quả từ các cuộc hội thoại, hỗ trợ cho việc lập chỉ mục và truy xuất thông tin trong các hệ thống trợ lý ảo.

Hy vọng những giải thích trên sẽ giúp Quốc hiểu rõ hơn về các loại prompt được sử dụng trong bài báo **LongMemEval**.

---

### 1. **Trích xuất Sự kiện có Dấu thời gian (Extracting Timestamped Events)**

**Mục tiêu**: Xác định và trích xuất các sự kiện liên quan đến người dùng từ tin nhắn, kèm theo ngày tháng xảy ra sự kiện đó.

**Cách thực hiện**: Sử dụng mô hình ngôn ngữ lớn (LLM) để phân tích danh sách tin nhắn của người dùng, xác định các sự kiện mà thời gian xảy ra có thể được suy luận hoặc được chỉ định rõ ràng. Mỗi sự kiện được trả về dưới dạng một đối tượng JSON với hai trường: "date" (ngày tháng theo định dạng YYYY/MM/DD) và "event" (mô tả sự kiện). Nếu không có sự kiện nào có thể xác định thời gian, trả về một danh sách trống.

**Ví dụ Prompt**:

```Plain
Bạn sẽ nhận được một danh sách các tin nhắn từ một người dùng gửi đến trợ lý AI, cùng với thời gian diễn ra cuộc trò chuyện. Hãy trích xuất tất cả các sự kiện liên quan đến người dùng mà ngày tháng của chúng được chỉ định hoặc có thể suy luận. Nếu không thể suy luận thời gian xảy ra của một sự kiện, không trích xuất sự kiện đó. Trả về các sự kiện dưới dạng một danh sách JSON, trong đó mỗi mục chứa hai trường: "date" và "event". Viết ngày tháng theo định dạng YYYY/MM/DD. Nếu không có sự kiện cụ thể nào, chỉ cần trả về một danh sách trống.
```



---

### 2. **Mở rộng Truy vấn (Query Expansion)**

**Mục tiêu**: Xác định khoảng thời gian có thể liên quan đến câu hỏi của người dùng về các sự kiện trước đó, nhằm cải thiện hiệu quả truy xuất thông tin.

**Cách thực hiện**: Khi nhận được một câu hỏi từ người dùng liên quan đến các sự kiện trong quá khứ, sử dụng LLM để suy luận khoảng thời gian mà các sự kiện trong khoảng đó có thể giúp trả lời câu hỏi. Kết quả được trả về dưới dạng một đối tượng JSON với hai trường: "start" và "end", biểu thị ngày bắt đầu và kết thúc của khoảng thời gian, theo định dạng YYYY/MM/DD. Nếu câu hỏi không có tham chiếu thời gian rõ ràng, trả về "N/A".

**Ví dụ Prompt**:

```Plain
Bạn sẽ nhận được một câu hỏi từ người dùng liên quan đến một số sự kiện trước đó, cùng với thời gian câu hỏi được đặt ra. Hãy suy luận một khoảng thời gian tiềm năng sao cho các sự kiện xảy ra trong khoảng này có thể giúp trả lời câu hỏi (bao gồm ngày bắt đầu và ngày kết thúc). Trả về một đối tượng JSON với hai trường: "start" và "end". Viết ngày tháng theo định dạng YYYY/MM/DD. Nếu câu hỏi không có bất kỳ tham chiếu thời gian nào, không cố gắng đoán khoảng thời gian. Thay vào đó, chỉ cần trả về "N/A".
```



---

**Lưu ý**: Các prompt trên được thiết kế để giúp hệ thống trợ lý AI xử lý và hiểu rõ hơn về ngữ cảnh thời gian trong các cuộc hội thoại, từ đó cải thiện khả năng truy xuất và trả lời chính xác hơn.

Hy vọng những giải thích trên giúp Quốc hiểu rõ hơn về cách sử dụng các prompt này trong việc trích xuất sự kiện và mở rộng truy vấn dựa trên thời gian.

```Plain
Có 1 ý tưởng nữa là dùng method: LLMs + Raptor =>

Đề xuất cải thiện:

1. Method cũ: Đem từng Round / Session đi Prompt để extract Summary/Fact/Keyphrase thì gặp vấn đề ngắn quá hoặc dài quá => Đề xuất: Chunking cho Session và prompt để lấy fact:

Chunking thì có nhiều method. Kỳ trước a Huy là chunking cho câu hỏi dạng QA. Của mình sẽ cải thiện hội thoại bằng Chunking trong Conversation.

1.1 Chunking theo LLMs chunk
1.2 Raptor Chunking: Chunk r

- Các Session được sử dụng phương pháp: LLMs Chunking + Raptor Chunking => Flat ra để thành các Embedding.

1.1 Raptor flat: Tức là đem cả Chunk ban đầu + LLMs Chunking + Summary Chunk đi Flat trải rộng ra và Rerank

1.2 Hierarchical Indexing - luồng 2 pha:

- Pha 1 (Coarse Retrieval):

Tạo embedding cho “summary” (hoặc keyphrase) của từng chunk.

Query → so sánh & lấy top-K chunk theo summary.

- Pha 2 (Fine Retrieval):

Với K chunk ở trên, ta đi sâu vào “fine index” (có facts/câu gốc).

- Lấy top-K’ final.

Reading: Đưa top-K’ này vào LLM để sinh câu trả lời.

2. Phương pháp cũ chỉ đề xuất đến việc: K=V+summary / K=V+fact / K=V+keyphrase

=> Đề xuất: Mỗi loại key có thế mạnh riêng (VD: summary giỏi match ý tổng quát, keyphrase giỏi match keyword cụ thể...). Ta embedding toàn bộ: V+summary, V+fact, V+keyphrase kèm với top K.
```

  

  

  

  

  

  

  

---

  

## 🚩 Vấn đề với phương pháp cũ (LongMemEval gốc)

  

1. **Granularity chưa tối ưu**:
    
    2. Việc trích xuất `summary`, `fact`, `keyphrase` từ **toàn bộ session** hoặc **round riêng lẻ** có thể gặp tình trạng:
        
        - Đoạn quá **ngắn** (không đủ ngữ cảnh để LLM trích xuất meaningful facts).
            
        - Đoạn quá **dài** (gây nhiễu thông tin, LLM không thể tóm tắt chính xác, dễ mất detail).
            
    3. Không có cách kiểm soát mức độ coherence hoặc topic shift trong session dài.
        
2. **Chỉ dùng 1 loại key duy nhất cho indexing**:
    
    2. `K = V + fact` hoặc `K = V + summary` là tốt, nhưng mỗi loại key có điểm mạnh khác nhau:
        
        - `summary`: tốt cho match semantic tổng thể.
            
        - `keyphrase`: bắt cụ thể keyword.
            
        - `fact`: truy xuất chính xác các entity, số liệu, mốc thời gian.
            
    3. Không tận dụng được hiệu ứng **ensemble giữa các loại key**.
        
3. **Lacking structure in indexing**:
    
    2. Indexing hiện tại là **flat** → không tận dụng tính chất "tầng" của văn bản hội thoại: đoạn – session – timeline.
        
    3. Thiếu khả năng điều hướng mượt mà giữa các mức khái quát (coarse) và chi tiết (fine).
        

  

---

  

## ✅ Giải pháp đề xuất: Kết hợp **LLMs + Raptor + Multi-Key Embedding + Hierarchical Indexing**

  

### **1. Conversation-Aware Chunking trước khi Extract**

  

#### ✂️ 1.1. LLM-based Chunking

  

- Dùng LLM để phân chia session thành các đoạn nhỏ (chunk) theo chuyển chủ đề, mục đích câu hỏi, hoặc hành vi người dùng.
    
- Lợi ích:
    
    - Tách được các segment theo topic.
        
    - Giữ được coherence bên trong mỗi chunk.
        

  

#### 🧱 1.2. Raptor Chunking

  

- Dùng **RAPTOR (recursive abstractive chunking)** để tạo cây phân cấp cho từng session.
    
- Mỗi node là một chunk hoặc summary của chunk con → có thể phục vụ **hierarchical retrieval**.
    

  

  

=> **Sau đó ta thu được: thay vì K = Session + fact thì của ta là: K = Session1.i + Fact (Với i là Session 1.i được Chunking nhỏ ra từ Session to ban đầu).**

  

![[Pasted image 20250322071142.png]]

---

  

### **2. Embedding: Flatten & Index**

  

#### 🧾 2.1 Raptor Flat Embedding

  

- Đưa từng chunk (ban đầu + LLM-chunked + summary chunk) vào embedding encoder.
    
- Tạo index **dạng phẳng (flat)**, có thể dùng Reranker để chọn top-K chunk có khả năng cao nhất.
    

  

#### 🧠 2.2 Hierarchical Indexing (2-phase Retrieval)

  

**Pha 1: Coarse Retrieval**

  

- Embed summary / keyphrase của chunk.
    
- Dùng query để so sánh, chọn Top-K chunk liên quan.
    

  

**Pha 2: Fine Retrieval**

  

- Với mỗi chunk đã chọn ở coarse stage → đi sâu vào level fine:
    
    - Embed lại các câu gốc / facts / sub-chunks.
        
    - Lấy top-K’ fine-grained memory units.
        

  

=> **Cuối cùng đưa vào LLM để đọc và trả lời (Reading stage).**

  

---

  

### **3. Multi-Key Embedding cho Indexing**

  

- Với mỗi chunk → tạo và embed song song:
    
    - `K1 = V + summary`
        
    - `K2 = V + fact`
        
    - `K3 = V + keyphrase`
        
- Kết hợp kết quả truy hồi từ các luồng (voting / weighted fusion / union-rerank).
    
- Lý do:
    
    - Summary bắt ngữ nghĩa chung.
        
    - Fact giúp reasoning logic.
        
    - Keyphrase giúp match keyword trong truy vấn cụ thể.
        

  

---

  

## 🔁 Tổng pipeline cải tiến

  

```Plain
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

  

---

  

## 📈 Lợi ích chính

  

- ✅ Giảm nhiễu, tối ưu granularity nhờ LLM chunking.
    
- ✅ Tận dụng tốt semantic, factual và keyword-level cues qua multi-key.
    
- ✅ Tăng khả năng truy hồi đúng với hierarchical search.
    
- ✅ Phù hợp để scale cho 500+ sessions như LongMemEvalM.
    

  

---

  

Nếu bạn cần mình mô phỏng pipeline này bằng sơ đồ (sequence hoặc flowchart), hoặc viết phần giải thích thành văn academic để ghép vào một paper / research blog, mình cũng làm được ngay nhé!

  

---

Dưới đây là toàn bộ **kịch bản thử nghiệm** (experimental settings) tương ứng với từng ý tưởng cải tiến mà Quốc đề xuất — được tổ chức theo dạng **ma trận thí nghiệm** để có thể dễ dàng triển khai thực nghiệm, đánh giá từng thành phần và kết hợp của pipeline.

  

---

  

## 🎯 **MỤC TIÊU THỬ NGHIỆM**

  

> Kiểm chứng các cải tiến về chunking, indexing, embedding, retrieval và reading strategy nhằm cải thiện hiệu quả của hệ thống long-term memory QA (ví dụ trên benchmark như LONGMEMEVAL).

  

---

  

## 🧪 **KỊCH BẢN THỬ NGHIỆM CHÍNH**

  

### 🔹 **I. Chunking Strategy**

  

|   |   |   |
|---|---|---|
|Mã|Tên phương pháp|Mô tả|
|C1|No Chunking (baseline)|Dùng cả session hoặc round làm value trực tiếp|
|C2|LLM-based Chunking|Phân chia đoạn theo chủ đề/ngữ nghĩa bằng LLM|
|C3|RAPTOR Chunking|Chunking dạng cây phân cấp theo RAPTOR|
|C4|LLM + RAPTOR Hybrid|Chunk theo LLM → dùng RAPTOR để tóm tắt từng chunk|

  

---

  

### 🔹 **II. Value Representation**

  

|   |   |   |
|---|---|---|
|Mã|Dạng value đầu vào|Mô tả|
|V1|Full Session|Không chia nhỏ, để nguyên session|
|V2|Round-based|Mỗi round là một value|
|V3|Chunked|Chunk theo chiến lược C2, C3, C4|
|V4|Summary|Tóm tắt của chunk hoặc session|
|V5|Fact|Fact trích từ chunk/session|

  

---

  

### 🔹 **III. Key Design (Indexing)**

  

|   |   |   |
|---|---|---|
|Mã|Tên thiết kế key|Mô tả|
|K1|K = V|Dùng raw value làm key|
|K2|K = fact|Key là facts đã trích|
|K3|K = summary|Key là summary|
|K4|K = V + fact|Nối fact vào value để tạo key|
|K5|K = V + summary|Nối summary vào value|
|K6|K = V + fact + summary + keyphrase|Multi-key (concat tất cả)|
|K7|Multi-path index|Tạo nhiều loại key riêng biệt, embed độc lập|

  

---

  

### 🔹 **IV. Retrieval Strategy**

  

|   |   |   |
|---|---|---|
|Mã|Phương pháp truy hồi|Mô tả|
|R1|Flat Retrieval|Retrieval đơn lớp, cosine / FAISS|
|R2|Coarse → Fine Retrieval (2-phase)|Truy xuất 2 pha: summary → fact|
|R3|Flat + Reranker|Retrieval sơ cấp rồi rerank bằng LLM|
|R4|Multi-path Fusion|Truy hồi theo từng key, rồi hợp kết quả (voting / union)|

  

---

  

### 🔹 **V. Reading Strategy**

  

|   |   |   |
|---|---|---|
|Mã|Kỹ thuật đọc kết quả|Mô tả|
|RS1|Direct Answer|Đưa chunk vào, yêu cầu LLM trả lời thẳng|
|RS2|Chain-of-Note (CoN)|Trích info trước rồi reasoning sau|
|RS3|JSON + CoN|Đưa input dạng JSON có cấu trúc, dùng CoN|
|RS4|CoT + CoN|Kết hợp chain-of-thought reasoning với CoN|

  

---

  

## ✅ **KẾT HỢP THỬ NGHIỆM GỢI Ý (FULL COMBO)**

  

|   |   |   |   |   |   |
|---|---|---|---|---|---|
|#|Chunking|Value|Key|Retrieval|Reading|
|1|C1|V1|K1|R1|RS1|
|2|C2|V3|K4|R2|RS3|
|3|C3|V3|K6|R2|RS2|
|4|C4|V3|K7|R4|RS3|
|5|C3|V5|K2|R3|RS1|
|6|C2|V3|K5|R1|RS2|
|7|C4|V3|K6|R3|RS4|

  

> 💡 _Bạn có thể chọn 3-5 combo để thử nghiệm A/B, đo các metric như:_
> 
>   
> 
> - **QA Accuracy (top-K)**
>     
> - **Recall@K, NDCG@K**
>     
> - **Inference time / latency**
>     
> - **Token count (input to reader)**
>     

  

---

  

## 📦 **Dữ liệu sử dụng**

  

- Dataset chính: `LONGMEMEVAL-S` và `LONGMEMEVAL-M`
    
- Có thể mở rộng thử nghiệm với `MemoryBank`, `PerLTQA` để kiểm tra độ generalize.
    

  

---

  

Nếu bạn cần mình giúp viết script pipeline chạy thử nghiệm hoặc dựng YAML config cho từng scenario để quản lý chạy bằng `Hydra`/`WandB` hoặc hệ thống ML pipeline khác thì cứ nói nhé. Cũng có thể dựng lại bảng này dưới dạng Notion template hoặc Google Sheet để dễ quản lý.

  

  

---

# SAU KHI HIỂU GỐC, BÀI BÁO GỐC, QUAY LẠI LANGGRAPH THÌ THẤY Ồ. RA LANGGRAPH NÓ ĐỀ XUẤT KIẾN TRÚC THÔI, CÒN CƠ BẢN THÌ LÀ BÀI BÁO ĐANG SEARCH NÀY

  

**LangGraph** là một **framework** được giới thiệu trong khóa học "Long-Term Agentic Memory with LangGraph" do Harrison Chase, Co-Founder và CEO của LangChain, giảng dạy. Khóa học này hướng dẫn cách xây dựng một **agent** với khả năng **ghi nhớ dài hạn**, cụ thể là trong việc quản lý email cá nhân.

  

**Điểm mới mà LangGraph đề cập đến**:

  

1. **Tích hợp ba loại memory trong agent**:
    
    2. **Semantic Memory**: Lưu trữ các **facts** về người dùng, như sở thích, thói quen, để sử dụng trong các tương tác sau này.
        
    3. **Episodic Memory**: Ghi nhớ các **tình huống cụ thể** đã xảy ra trong quá khứ, giúp agent hiểu ngữ cảnh và cải thiện phản hồi.
        
    4. **Procedural Memory**: Lưu trữ các **hướng dẫn và quy trình** mà agent cần tuân theo, giúp tối ưu hóa hành vi dựa trên phản hồi.
        

  

  

---

https://github.com/DoanNgocCuong/MiniProj_RAG3_RAG6_LegalChatbot_16032025

  

---

# Hiểu sâu hơn về Datase:

  

  

LongMemEval là một bộ dữ liệu toàn diện, được thiết kế để đánh giá khả năng ghi nhớ dài hạn của các trợ lý trò chuyện. Bộ dữ liệu này bao gồm 500 câu hỏi chất lượng cao, tập trung vào năm khả năng cốt lõi:

  

1. **Trích xuất thông tin (Information Extraction):** Khả năng nhớ lại thông tin cụ thể từ lịch sử tương tác dài, bao gồm cả chi tiết do người dùng hoặc trợ lý cung cấp.
    

2. **Lý luận đa phiên (Multi-Session Reasoning):** Khả năng tổng hợp thông tin từ nhiều phiên trò chuyện để trả lời các câu hỏi phức tạp yêu cầu sự tổng hợp và so sánh.
    

3. **Cập nhật kiến thức (Knowledge Updates):** Khả năng nhận biết và cập nhật thông tin cá nhân của người dùng theo thời gian.
    

4. **Lý luận thời gian (Temporal Reasoning):** Nhận thức về các khía cạnh thời gian của thông tin người dùng, bao gồm cả thời gian được đề cập rõ ràng và siêu dữ liệu thời gian trong các tương tác.
    

5. **Từ chối trả lời (Abstention):** Khả năng từ chối trả lời các câu hỏi liên quan đến thông tin không được đề cập trong lịch sử tương tác.
    

  

Lấy cảm hứng từ bài kiểm tra "tìm kim trong đống cỏ khô", LongMemEval sử dụng một quy trình kiểm soát thuộc tính để tạo ra lịch sử trò chuyện mạch lạc, có thể mở rộng và được đánh dấu thời gian cho mỗi câu hỏi. Hệ thống trò chuyện cần phân tích các tương tác động để ghi nhớ và trả lời câu hỏi sau khi tất cả các phiên tương tác đã diễn ra.

  

**Cấu trúc Bộ Dữ Liệu:**

  

Bộ dữ liệu bao gồm ba tệp chính:

  

1. **longmemeval_s.json:** Mỗi lịch sử trò chuyện tiêu thụ khoảng 115.000 token (~40 phiên lịch sử).
    

2. **longmemeval_m.json:** Mỗi lịch sử trò chuyện chứa khoảng 500 phiên.
    

3. **longmemeval_oracle.json:** Chỉ bao gồm các phiên chứa bằng chứng cần thiết.
    

  

Mỗi tệp chứa 500 trường hợp đánh giá, mỗi trường hợp bao gồm các trường:

  

- **question_id:** ID duy nhất cho mỗi câu hỏi.
    

- **question_type:** Loại câu hỏi, như single-session-user, single-session-assistant, single-session-preference, temporal-reasoning, knowledge-update, và multi-session. Nếu question_id kết thúc bằng _abs, đó là câu hỏi từ chối trả lời.
    

- **question:** Nội dung câu hỏi.
    

- **answer:** Câu trả lời mong đợi từ mô hình.
    

- **question_date:** Ngày của câu hỏi.
    

- **haystack_session_ids:** Danh sách ID của các phiên lịch sử (sắp xếp theo thời gian).
    

- **haystack_dates:** Danh sách các mốc thời gian của các phiên lịch sử.
    

- **haystack_sessions:** Danh sách nội dung thực tế của các phiên trò chuyện giữa người dùng và trợ lý. Mỗi phiên là một danh sách các lượt trao đổi, mỗi lượt có định dạng {"role": user/assistant, "content": nội dung tin nhắn}. Đối với các lượt chứa bằng chứng cần thiết, có thêm trường has_answer: true.
    

- **answer_session_ids:** Danh sách ID của các phiên chứa bằng chứng, dùng để đánh giá độ chính xác của việc nhớ lại ở cấp độ phiên.
    

  

**Thiết lập Môi Trường:**

  

Để sử dụng bộ dữ liệu, bạn có thể tải xuống từ [Hugging Face](https://huggingface.co/datasets/xiaowu0162/longmemeval) và giải nén vào thư mục `data/`. Khuyến nghị sử dụng môi trường conda để cài đặt các yêu cầu cần thiết:

  

```Bash
conda create -n longmemeval python=3.9
conda activate longmemeval
pip install -r requirements-full.txt
```

  



  

**Đánh Giá Hệ Thống:**

  

Để kiểm tra hệ thống của bạn trên LongMemEval, bạn có thể sử dụng các tập lệnh đánh giá được cung cấp. Lưu đầu ra của hệ thống vào tệp JSONL với mỗi dòng chứa hai trường: `question_id` và `hypothesis`. Sau đó, chạy tập lệnh đánh giá:

  

```Bash
export OPENAI_API_KEY=YOUR_API_KEY
cd src/evaluation
python3 evaluate_qa.py gpt-4o your_hypothesis_file ../../data/longmemeval_oracle.json
```

  



  

Tập lệnh này sẽ lưu nhật ký đánh giá vào tệp `[your_hypothesis_file].log`. Bạn có thể tổng hợp các điểm số từ nhật ký bằng lệnh:

  

```Bash
python3 print_qa_metrics.py gpt-4o your_hypothesis_file.log ../../data/longmemeval_oracle.json
```

  



  

**Tạo Lịch Sử Trò Chuyện Tùy Chỉnh:**

  

LongMemEval hỗ trợ biên soạn lịch sử trò chuyện với độ dài tùy ý cho mỗi trường hợp câu hỏi, cho phép bạn dễ dàng tăng độ khó. Để tạo lịch sử tùy chỉnh, bạn có thể làm theo định dạng trong `2_questions` và `6_session_cache` để tạo câu hỏi và các phiên bằng chứng, sau đó chạy tập lệnh `sample_haystack_and_timestamp.py` với các tham số phù hợp.

  

**Chạy Thử Nghiệm Hệ Thống Ghi Nhớ:**

  

Chúng tôi cung cấp mã thử nghiệm cho việc truy xuất bộ nhớ và tạo câu trả lời có hỗ trợ truy xuất dưới các thư mục `src/retrieval

  

  

---

Long-TermMemoryMethods Toequipchatassistantswithlong-termmemorycapabilities, three major techniques are commonly explored. The first approach involves directly adapting LLMs to process extensive history information as long-context inputs (Beltagy et al., 2020; Kitaev et al., 2020; Fu et al., 2024; An et al., 2024). While this method avoids the need for complex architectures, it is inefficient and susceptible to the “lost-in-the-middle” phenomenon, where the ability of LLMs to utilize contextual information weakens as the input length grows (Shi et al., 2023; Liu et al., 2024). A second line of research integrates differentiable memory modules into language models, proposing specialized architectural designs and training strategies to enhance memory capabilities (Weston et al., 2014; Wu et al., 2022; Zhong et al., 2022; Wang et al., 2023). Lastly, several studies approach long-term memory from the perspective of context compression, developing techniques 3 Published as a conference paper at ICLR 2025 to condense lengthy histories into compact representations, whether in the form of LLM internal representations (Mu et al., 2023; Chevalier et al., 2023), discrete tokens (Jiang et al., 2023; Xu et al., 2024), or retrievable text segments via retrieval-augmented generation (RAG, Shi et al. (2024); Wang et al. (2023); Sarthi et al. (2024); Chen et al. (2023a); Guti´ errez et al. (2024)). Although LONGMEMEVAL can evaluate any memory system, we will take an online context compression perspective, where each history interaction session is sequentially processed, stored, and accessed on-demand through indexing and retrieval mechanisms (§4). This formulation aligns with current literature (Zhong et al., 2024; Guti´ errez et al., 2024) and commercial systems (OpenAI, 2024; Coze, 2024). Its plug-and-play nature also facilitates the integration into existing chat assistant systems

  

  

Dưới đây là bản dịch tiếng Việt đoạn văn bạn cung cấp:

  

---

  

### **Các phương pháp trí nhớ dài hạn (Long-Term Memory Methods)**

  

Để trang bị khả năng ghi nhớ dài hạn cho các trợ lý hội thoại, hiện có ba kỹ thuật chính thường được nghiên cứu:

  

1. **Phương pháp thứ nhất** là điều chỉnh trực tiếp các mô hình ngôn ngữ lớn (LLMs) để xử lý lượng lớn thông tin lịch sử dưới dạng đầu vào dài (long-context input) _(Beltagy et al., 2020; Kitaev et al., 2020; Fu et al., 2024; An et al., 2024)_. Phương pháp này giúp tránh việc phải thiết kế kiến trúc phức tạp, tuy nhiên lại **kém hiệu quả** và dễ gặp hiện tượng **"mất thông tin ở giữa" (lost-in-the-middle)** – khi mà khả năng của LLM trong việc tận dụng thông tin ngữ cảnh suy giảm theo độ dài đầu vào tăng lên _(Shi et al., 2023; Liu et al., 2024)_.
    

2. **Hướng nghiên cứu thứ hai** là tích hợp các **module bộ nhớ phân biệt được (differentiable memory modules)** vào trong mô hình ngôn ngữ. Các nghiên cứu này đề xuất các thiết kế kiến trúc chuyên biệt và chiến lược huấn luyện nhằm tăng cường khả năng ghi nhớ của mô hình _(Weston et al., 2014; Wu et al., 2022; Zhong et al., 2022; Wang et al., 2023)_.
    

3. **Cuối cùng**, nhiều nghiên cứu tiếp cận trí nhớ dài hạn từ góc độ **nén ngữ cảnh (context compression)**, phát triển các kỹ thuật nhằm **tinh gọn lịch sử hội thoại dài** thành các biểu diễn nhỏ gọn hơn – có thể dưới dạng biểu diễn nội tại trong LLM _(Mu et al., 2023; Chevalier et al., 2023)_, các token rời rạc _(Jiang et al., 2023; Xu et al., 2024)_, hoặc các đoạn văn bản có thể truy xuất được thông qua kỹ thuật sinh có hỗ trợ truy hồi (Retrieval-Augmented Generation - RAG) _(Shi et al., 2024; Wang et al., 2023; Sarthi et al., 2024; Chen et al., 2023a; Gutiérrez et al., 2024)_.
    

  

Mặc dù **LONGMEMEVAL** có thể được dùng để đánh giá bất kỳ hệ thống trí nhớ nào,

trong bài này chúng tôi chọn cách tiếp cận theo hướng **nén ngữ cảnh trực tuyến (online context compression)**,

nơi mà mỗi phiên tương tác trong lịch sử sẽ được **xử lý tuần tự, lưu trữ và truy xuất theo yêu cầu** thông qua các cơ chế đánh chỉ mục (indexing) và truy hồi (retrieval) (§4).

  

Cách tiếp cận này phù hợp với các công trình hiện tại _(Zhong et al., 2024; Gutiérrez et al., 2024)_

cũng như các hệ thống thương mại như **OpenAI (2024)** và **Coze (2024)**.

Đặc biệt, nhờ vào tính **"plug-and-play"** (cắm vào là chạy), phương pháp này có thể dễ dàng tích hợp vào các hệ thống trợ lý hội thoại hiện có.

  

---

  

Nếu bạn muốn mình tóm lại thành bảng so sánh 3 hướng tiếp cận hoặc biểu đồ sơ đồ hóa thì mình có thể vẽ liền nhé!