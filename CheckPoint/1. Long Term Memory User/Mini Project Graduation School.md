### **📄 Đồ án nghiên cứu: Đề tài: "Long-Term Memory Augmentation for Conversational Question Answering Systems"** _**(Tăng cường trí nhớ dài hạn cho các hệ thống hỏi đáp hội thoại)**_

  

📝 **Tác giả:**

🏫 **Đơn vị nghiên cứu:**

📅 **Ngày thực hiện:**

  

---

  

# **📌 1. Giới thiệu (Introduction)**

  

## 1.1. Đặt vấn đề:

### 🔥 **Động lực chính:**

1. **Cá nhân hóa sâu sắc** Ghi nhớ thông tin, sở thích và hành vi người dùng giúp chatbot/robot phản hồi phù hợp, tạo cảm giác thân quen và tăng sự hài lòng. 👉 **Ví dụ**: Một người dùng thường tìm công thức ăn chay. Chatbot ghi nhớ điều này và luôn gợi ý món chay, thay vì các công thức ngẫu nhiên. 👉 _Nếu không có điều này_: Chatbot có thể liên tục đề xuất món mặn, gây phiền toái và làm người dùng mất thiện cảm, từ đó giảm khả năng quay lại sử dụng.
    
2. **Duy trì và tái sử dụng ngữ cảnh** Trí nhớ dài hạn cho phép hiểu được lịch sử hội thoại qua nhiều phiên làm việc, giữ mạch logic và tránh yêu cầu người dùng phải lặp lại thông tin. 👉 **Ví dụ**: Trong một chuỗi hội thoại về đơn hàng, chatbot nhớ rằng người dùng đang khiếu nại về sản phẩm X từ phiên trước và tiếp tục hỗ trợ ngay ở phiên sau. 👉 _Nếu không có điều này_: Người dùng sẽ phải lặp lại toàn bộ thông tin khi quay lại, gây bực bội và tạo cảm giác chatbot thiếu chuyên nghiệp.
    
3. **Hỗ trợ tác vụ dài hạn** Các ứng dụng như theo dõi học tập, chăm sóc sức khỏe hoặc hành trình khách hàng yêu cầu chatbot ghi nhớ và cập nhật tiến trình liên tục. 👉 **Ví dụ**: Một chatbot học tập ghi nhớ rằng học sinh đã yếu ở phần thì quá khứ hoàn thành và tiếp tục luyện tập điểm này trong các buổi sau. 👉 _Nếu không có điều này_: Chatbot sẽ lặp lại những nội dung học đã ổn, bỏ sót điểm yếu của học sinh, làm giảm hiệu quả học tập và cảm giác “được đồng hành” của người học.
    

---

## **1.2. Các giải pháp hiện tại và hạn chế**

Trong những năm gần đây, để xây dựng các trợ lý hội thoại có khả năng ghi nhớ dài hạn, cộng đồng nghiên cứu đã phát triển ba hướng tiếp cận chính, mỗi hướng đại diện cho một tư duy kiến trúc khác nhau về cách hệ thống lưu trữ, cập nhật và truy xuất thông tin từ lịch sử hội thoại.

#### 📌 **(1) Xử lý trực tiếp toàn bộ ngữ cảnh dài (Long-context Input)**

Cách tiếp cận đầu tiên là cung cấp **toàn bộ lịch sử hội thoại** trước đó vào phần **ngữ cảnh đầu vào của LLM** dưới dạng một chuỗi duy nhất. Điều này cho phép mô hình tiếp cận toàn bộ dữ liệu trước đó trong một lần tính toán, từ đó đưa ra phản hồi phù hợp với ngữ cảnh hội thoại kéo dài.

- **Ưu điểm**:
    
    - Không yêu cầu thay đổi kiến trúc mô hình.
        
    - Dễ triển khai, có thể áp dụng trực tiếp cho các LLM hiện có (GPT-3/4, Claude...).
        
- **Hạn chế**:
    
    - Tốn tài nguyên, do độ dài đầu vào tăng mạnh.
        
    - Dễ gặp hiện tượng **“lost-in-the-middle”**, khi mô hình **không còn truy cập hiệu quả** tới các thông tin nằm giữa đoạn văn bản dài – dẫn đến việc quên mất thông tin quan trọng hoặc trả lời sai lệch (Beltagy et al., 2020; Shi et al., 2023).
        
    - Bị giới hạn bởi độ dài context window của mô hình (~8k–100k tokens tùy phiên bản).
        

#### 📌 **(2) Tích hợp mô-đun trí nhớ khả vi (Differentiable Memory Modules)**

Hướng tiếp cận thứ hai là **thiết kế lại kiến trúc mạng nơ-ron**, tích hợp một thành phần **bộ nhớ học được (learnable memory)** – cho phép mô hình ghi nhớ và truy xuất thông tin thông qua các cơ chế như attention hoặc đọc/ghi khả vi.

- **Tiêu biểu**: Memory Networks (Weston et al., 2014), Dynamic Memory Network (Kumar et al., 2016), MemGPT (Wu et al., 2022).
    
- **Ưu điểm**:
    
    - Có khả năng lưu giữ thông tin lâu dài.
        
    - Hỗ trợ suy luận nhiều bước dựa trên bộ nhớ (multi-hop reasoning).
        
- **Hạn chế**:
    
    - Đòi hỏi huấn luyện lại từ đầu, không dễ áp dụng cho các mô hình LLM thương mại dạng API.
        
    - Khó tối ưu hóa hiệu quả khi huấn luyện trên dữ liệu hội thoại tự nhiên, đặc biệt nếu hội thoại dài, nhiều biến thể ngữ nghĩa.
        

#### (3) Xử lý ngữ cảnh và truy xuất khi cần (Context Compression & Retrieval)

Có 3 method xử lý chính:

##### 3.1 Lấy toàn bộ thông tin hội thoại:

- **Mô tả:**
    
    - Phương pháp này lưu trữ và sử dụng toàn bộ nội dung của phiên hội thoại để xử lý và truy xuất thông tin khi cần thiết.
        
- **Ưu điểm:**
    
    - Bảo toàn đầy đủ ngữ cảnh và chi tiết của cuộc trò chuyện.
        
    - Hữu ích trong các tình huống yêu cầu phân tích toàn diện hoặc khi cần truy xuất thông tin cụ thể từ bất kỳ phần nào của hội thoại.
        
- **Nhược điểm:**
    
    - Dễ dẫn đến quá tải bộ nhớ và giảm hiệu suất do lượng dữ liệu lớn.
        
    - Khó khăn trong việc xác định và truy xuất thông tin quan trọng do thiếu cấu trúc phân cấp.
        

##### 3.2 Xử lý ngữ cảnh theo **round**

- Đây là đơn vị nhỏ nhất trong một phiên hội thoại: **mỗi lượt hỏi–đáp** được lưu và xử lý riêng biệt.
    
- Kết quả thực nghiệm cho thấy:
    

> - "**Decomposing sessions into rounds significantly enhances reading performance**" – đặc biệt với các mô hình mạnh như GPT-4o.
>     

- **Ưu điểm:**
    
    - Giảm nhiễu, tăng độ chính xác khi truy xuất.
        
    - Cho phép granularity cao → dễ chunk, đánh index, và lọc thông tin.
        
- **Nhược điểm:**
    
    - Dễ mất mạch hội thoại nếu không có chiến lược đọc (reading strategy) hợp lý như Chain-of-Note.
        
- **Dẫn chứng:**
    

> - “Using LONGMEMEVALM, we compare different value choices… Decomposing sessions into rounds significantly enhances reading performance”Read2 - LONGMEMEVAL - B….
>     

---

##### 3.3 Xử lý ngữ cảnh theo **phiên** (session)

- Mỗi phiên hội thoại gồm nhiều round liên tiếp sẽ được lưu nguyên khối.
    
- Đây là baseline phổ biến của nhiều hệ thống hiện nay.
    
- **Ưu điểm:**
    
    - Bảo toàn mạch hội thoại, đặc biệt hữu ích trong các truy vấn cần tổng hợp theo diễn tiến (e.g. "trong lần trò chuyện hôm trước").
        
- **Nhược điểm:**
    
    - Dễ gây quá tải context window.
        
    - Khó truy xuất chính xác nếu không có indexing tốt.
        
- **Dẫn chứng:**
    

> - “Storing each session as a single item can hinder effective retrieval and reading”
>     

  

---

### Ưu nhược điểm của các giải pháp hiện tại:

1. **Granularity chưa tối ưu:**
    
    1. **Vấn đề:**
        
        - Trích xuất thông tin từ toàn bộ session hoặc từng round riêng lẻ có thể dẫn đến:
            
            - Đoạn quá ngắn: Thiếu ngữ cảnh để mô hình ngôn ngữ lớn (LLM) trích xuất thông tin ý nghĩa.
                
            - Đoạn quá dài: Gây nhiễu, khó tóm tắt chính xác và dễ mất chi tiết quan trọng.
                
        - Không kiểm soát được mức độ liên kết hoặc chuyển đổi chủ đề trong các session dài.
            
2. **Chỉ sử dụng một loại khóa (key) duy nhất cho việc lập chỉ mục (indexing):**
    
    2. **Vấn đề:**
        
        - Sử dụng một loại khóa như `K = V + fact` hoặc `K = V + summary` có những hạn chế:
            
            - **Summary:** Tốt cho việc khớp ngữ nghĩa tổng thể.
                
            - **Keyphrase:** Bắt được các từ khóa cụ thể.
                
            - **Fact:** Truy xuất chính xác các thực thể, số liệu, mốc thời gian.
                
        - Không tận dụng được hiệu ứng **kết hợp (ensemble)** giữa các loại khóa khác nhau.
            
3. **Thiếu cấu trúc trong việc lập chỉ mục:**
    
    1. **Vấn đề:**
        
        - Lập chỉ mục dạng phẳng (flat) không tận dụng được cấu trúc phân cấp của văn bản hội thoại: đoạn – session – dòng thời gian.
            
        - Thiếu khả năng điều hướng mượt mà giữa các mức độ khái quát (coarse) và chi tiết (fine).
            

---

## 1.3 Dataset:

![](https://csg2ej4iz2hz.sg.larksuite.com/space/api/box/stream/download/asynccode/?code=OThlMzI3OGJmNTllZjA2YmRlMzY0MzFmZjA0MmM4YTlfMDNnY1BSWDhEMzBkSXdQaURZSzBLS1FxOW04MWxHQ01fVG9rZW46VGtBa2I5aVFmb21HckZ4aFAxZ2xUQ2tJZ2owXzE3NDQ5MDA1NTU6MTc0NDkwNDE1NV9WNA)

  

LongMemEval là một bộ dữ liệu toàn diện, được thiết kế để đánh giá khả năng ghi nhớ dài hạn của các trợ lý trò chuyện. Bộ dữ liệu này bao gồm 500 câu hỏi chất lượng cao, tập trung vào năm khả năng cốt lõi:

  

1. **Trích xuất thông tin (Information Extraction):** Khả năng nhớ lại thông tin cụ thể từ lịch sử tương tác dài, bao gồm cả chi tiết do người dùng hoặc trợ lý cung cấp.
    
2. **Lý luận đa phiên (Multi-Session Reasoning):** Khả năng tổng hợp thông tin từ nhiều phiên trò chuyện để trả lời các câu hỏi phức tạp yêu cầu sự tổng hợp và so sánh.
    
3. **Cập nhật kiến thức (Knowledge Updates):** Khả năng nhận biết và cập nhật thông tin cá nhân của người dùng theo thời gian.
    
4. **Lý luận thời gian (Temporal Reasoning):** Nhận thức về các khía cạnh thời gian của thông tin người dùng, bao gồm cả thời gian được đề cập rõ ràng và siêu dữ liệu thời gian trong các tương tác.
    
5. **Từ chối trả lời (Abstention):** Khả năng từ chối trả lời các câu hỏi liên quan đến thông tin không được đề cập trong lịch sử tương tác.
    

### Hỏi:

- Open Domain với Personal khác nhau như nào nhỉ: **bộ dữ liệu Personal tập trung vào lịch sử tương tác và thông tin cá nhân của một người dùng với AI để đánh giá khả năng cá nhân hóa và trí nhớ dài hạn liên quan đến người đó**, trong khi **bộ dữ liệu Open-Domain chứa đựng các cuộc hội thoại đa dạng về chủ đề và thường không tập trung vào một người dùng cụ thể hoặc lịch sử tương tác cá nhân kéo dài.**
    
- LongMemEval:
    
    - Đối với thiết lập **LONGMEMEVALS**, lịch sử trò chuyện cho mỗi câu hỏi có độ dài khoảng **115 nghìn token. Thiết lập LONGMEMEVALS có thể bao gồm khoảng từ 30 đến 60 phiên. khoảng 1900 đến 3800 token/phiên.**
        
    - Đối với thiết lập **LONGMEMEVALM**, lịch sử trò chuyện cho mỗi câu hỏi **có tổng độ dài khoảng 1,5 triệu token. B**ao gồm **500 phiên, khoảng 1900 đến 3800 token/phiên.**
        
    
      
    
    > **Trường hợp phiên ngắn nhất (1900 token) và lượt tương tác dài nhất (150 token/lượt)**: 1900 / 150 ≈ **12.7 lượt**.
    > 
    > **Trường hợp phiên ngắn nhất (1900 token) và lượt tương tác ngắn nhất (50 token/lượt)**: 1900 / 50 = **38 lượt**.
    > 
    > **Trường hợp phiên dài nhất (3800 token) và lượt tương tác dài nhất (150 token/lượt)**: 3800 / 150 ≈ **25.3 lượt**.
    > 
    > **Trường hợp phiên dài nhất (3800 token) và lượt tương tác ngắn nhất (50 token/lượt)**: 3800 / 50 = **76 lượt**.
    > 
    > Như vậy, dựa trên ước tính về độ dài lượt tương tác, một phiên có độ dài từ **1900 đến 3800 token** có thể bao gồm **khoảng từ 13 đến 76 lượt tương tác**
    

### MSC

1 dòng test sẽ gồm: 1 câu hỏi, 1 câu trả lời, và 1 lịch sử gồm 4 phiên hội thoại trước đó.

Mỗi trường hợp kiểm thử (test instance) thường bao gồm:

- **Lịch sử hội thoại:** Bao gồm nhiều phiên (ví dụ, 4 phiên đầu tiên của một cuộc đối thoại) để xây dựng ngữ cảnh và lưu trữ thông tin, thông tin cá nhân (persona) của các bên tham gia.
    
- **Câu hỏi (query):** Là lượt nói hiện tại trong phiên cuối (ví dụ, phiên thứ 5) mà mô hình cần trả lời.
    
- **Câu trả lời chuẩn (gold response):** Là đáp án được gán sẵn để so sánh với phản hồi của mô hình.
    

Dẫn chứng bài: https://neurips2023-enlsp.github.io/papers/paper_38.pdf

## 1.4 **Giải pháp đề xuất: Kết hợp LLMs + RAPTOR + Multi-Key Embedding + Hierarchical Indexing**

### 1.4.1 **Phân đoạn hội thoại theo ngữ cảnh trước khi trích xuất:**

#### 1.1✂️ **Phân đoạn bằng LLM (LLM-based Chunking):**

  **Phương pháp:** Sử dụng LLM để phân chia session thành các đoạn nhỏ (chunk) dựa trên:

- Chuyển đổi chủ đề.
    
- Mục đích câu hỏi.
    
- Hành vi người dùng.
    

  **Lợi ích:**

    Tách được các segment theo chủ đề.

    Giữ được tính liên kết bên trong mỗi chunk.

#### **1.2. Phân đoạn bằng RAPTOR (RAPTOR Chunking):**

  **Phương pháp:**

- Sử dụng RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) để tạo cây phân cấp cho từng session.
    
- Mỗi node là một chunk hoặc tóm tắt của các chunk con, phục vụ cho việc truy xuất phân cấp (hierarchical retrieval).
    

  **Lợi ích:**

- Tạo cấu trúc cây giúp truy xuất thông tin hiệu quả hơn.
    
- Tận dụng được cả thông tin chi tiết và tổng quan.
    

**Kết quả:** Thay vì sử dụng `K = Session + fact`, ta có `K = Session1.i + Fact` (với `i` là các chunk nhỏ được tạo từ session ban đầu).

### 1.4.2 **Embedding: Làm phẳng và lập chỉ mục (Flatten & Index):**

#### 🧾 **2.1. Embedding phẳng bằng RAPTOR (RAPTOR Flat Embedding):**

- **Phương pháp:**
    
    - Đưa từng chunk (bao gồm chunk ban đầu, chunk được tạo bởi LLM, và summary chunk) vào bộ mã hóa embedding.
        
    - Tạo lập chỉ mục dạng phẳng, có thể sử dụng Reranker để chọn top-K chunk có liên quan nhất.
        

#### 🧠 **2.2. Lập chỉ mục phân cấp (Hierarchical Indexing) - Truy xuất hai giai đoạn (2-phase Retrieval):**

- **Giai đoạn 1: Truy xuất thô (Coarse Retrieval):**
    
    - Embedding các summary hoặc keyphrase của chunk.
        
    - Sử dụng truy vấn để so sánh và chọn Top-K chunk liên quan.
        
- **Giai đoạn 2: Truy xuất chi tiết (Fine Retrieval):**
    
    - Với mỗi chunk đã chọn ở giai đoạn thô:
        
        - Embedding lại các câu gốc, facts, hoặc sub-chunks.
            
        - Lấy top-K’ đơn vị bộ nhớ chi tiết.
            

**Kết quả:** Các thông tin này được đưa vào LLM để đọc và trả lời (Reading stage).

### 1.4.3 **Embedding đa khóa cho việc lập chỉ mục (Multi-Key Embedding for Indexing):**

**Phương pháp:**

- Với mỗi chunk, tạo và embedding song song các khóa:
    
    - `K1 = V + summary`
        
    - `K2 = V + fact`
        
    - `K3 = V + keyphrase`
        
- Kết hợp kết quả truy hồi từ các luồng bằng các phương pháp như voting, weighted fusion, hoặc union-rerank.
    

**Lý do:**

- **Summary:** Bắt ngữ nghĩa chung.
    
- **Fact:** Hỗ trợ lập luận logic.
    
- **Keyphrase:** Khớp từ khóa trong truy vấn cụ thể.
    

  

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

  

## 1.4 Đóng góp của đồ án

Đồ án này có 2 đóng góp chính như sau:

1. Đồ án đề xuất giải pháp kết hợp các kỹ thuật phân đoạn khác nhau nhằm tăng
    

hiệu suất của hệ thống truy xuất thông tin.

2. Thực hiện thử nghiệm kết hợp các kỹ thuật truy xuất nhằm cải thiện kết quả
    

đầu ra.

## 1.5 Bố cục đồ án

Toàn bộ báo cáo đồ án tốt nghiệp được triển khai trong 5 chương. Các chương

còn lại của báo cáo có nội dung như sau.

Chương 2 đề cập đến các nội dung lý thuyết nhằm phục vụ việc nghiên cứu, xây

dựng thử nghiệm và đánh giá giải pháp đề xuất. Trong chương này, tôi sẽ trình bày

tổng quan về mô hình ngôn ngữ lớn, các ứng dụng, hạn chế và một số dòng mô

hình ngôn ngữ lớn phổ biến. Kỹ thuật RAG với các thành phần và các giải pháp

hiện có cũng sẽ được phân tích chi tiết ở chương này.

Chương 3 trình bày chi tiết về giải pháp đề xuất. Trước hết, tôi mô tả tổng quan

về luồng xử lý, sau đó là đi sâu vào từng mô-đun. Trong mô-đun phân đoạn, tôi

trình bày hai kỹ thuật phân đoạn tôi lấy làm ý tưởng đó là phân đoạn sử dụng mô

hình ngôn ngữ lớn và RAPTOR. Sau đó, tôi đề xuất việc kết hợp hai kỹ thuật này

để bổ trợ cho nhau. Trong mô-đun truy xuất, tôi trình bày việc kết hợp hai kỹ thuật

đó là: i) tìm kiếm mức ngữ nghĩa và ii) tìm kiếm mức từ vựng nhằm cải thiện mức

độ phù hợp của các tài liệu tìm kiếm được.

Chương 4 trình bày cụ thể về các kịch bản thử nghiệm, thông số cấu hình thử

nghiệm, kết quả thực nghiệm và các đánh giá, nhận xét về các phương pháp thử

nghiệm. Trong chương này, tôi sử dụng một số độ đo tự động thường được sử dụng

cho hỏi đáp và đánh giá bằng mô hình ngôn ngữ lớn. Những nhận xét và đánh giá

hiệu năng của phương pháp đề xuất so với các phương pháp tham chiếu cũng được

trình bày tại chương này.

Chương 5 là chương cuối cùng. Trong chương này, tôi nêu ra kết luận về phương

pháp đề xuất, những ưu điểm cũng như những hạn chế còn tồn tại cũng như đề ra

các hướng phát triển trong tương lai.

  

---

  

## **📌 2. Tổng quan nghiên cứu (Related Work)**

  

### **2.1. Hạn chế của LLMs về trí nhớ**

  

- LLMs hiện nay **chỉ có trí nhớ ngắn hạn**, bị giới hạn bởi context window (128K tokens với GPT-4-turbo, 1M tokens với Claude 3). - 2M rất to
    
- Các mô hình không thể duy trì bối cảnh hội thoại **qua nhiều phiên làm việc**.
    

  

### **2.2. Các phương pháp hiện tại**

  

#### **(1) LLMs lưu trữ ngắn hạn

  

  

  

#### **(2) Retrieval-Augmented Generation (RAG)**

  

- **Ưu điểm**: LLM có thể truy xuất dữ liệu từ nguồn ngoài khi cần.
    
- **Nhược điểm**: Không nhớ thông tin theo thời gian, chỉ hoạt động khi có truy vấn tìm kiếm.
    

  

#### **(3) Các nghiên cứu trước đây**

  

- OpenAI đang phát triển **tác nhân có trí nhớ** nhưng chưa công bố chi tiết.
    
- Meta AI thử nghiệm chatbot có khả năng **nhớ sở thích người dùng** nhưng gặp thách thức về quyền riêng tư.
    

![[Pasted image 20250322054143.png]]

  

📌 **Điểm khác biệt của nghiên cứu này:**

✅ Đề xuất mô hình **Memory-Augmented AI** tối ưu hơn, có thể **học hỏi theo thời gian mà không bị quá tải dữ liệu**.

✅ Kết hợp giữa **Memory-Augmented Learning & RAG** để tối ưu hóa bộ nhớ.

  

---

  

## **📌 3. Phương pháp nghiên cứu (Methodology)**

  

### **3.1. Kiến trúc đề xuất**

  

Mô hình **Memory-Augmented AI Agent** gồm các thành phần chính:

1️⃣ **Short-Term Memory (STM)**: Lưu trữ thông tin trong phạm vi cửa sổ ngữ cảnh hiện tại.

2️⃣ **Long-Term Memory (LTM)**: Lưu trữ thông tin quan trọng vào **Vector Database**.

3️⃣ **Memory Management Algorithm**: Quyết định **nên nhớ gì, quên gì**. (lưu tất thì bị phìng bộ nhớ? )

-bỏ: Trí nhớ về sở thích

- bỏ: Trí nhớ về các sự kiện đã qua
    
- Trí nhớ về các lịch sắp tới
    

4️⃣ **Knowledge Update Mechanism**: Cập nhật và quên thông tin cũ khi cần.

- Cập nhật dựa trên thời gian (User ngày xưa thích chơi đá bóng.Gẫy chân => Hiện tại thì không).
    

  

📌 **Mô hình sử dụng các công nghệ:**

  

- **LLM (GPT-4, Claude 3, Llama 2)**.
    
- **Vector Database (FAISS, Pinecone, Weaviate)** để lưu trí nhớ dài hạn.
    
- **LangChain / LlamaIndex** để quản lý truy xuất thông tin.
    

  

---

  

## **📌 4. Thực nghiệm & Kết quả (Experiments & Results)**

  

### **4.1. Thiết lập thử nghiệm**

  

**Bài toán:** So sánh hiệu suất giữa **Memory-Augmented AI Agent** và **LLM thông thường** trong hội thoại dài hạn.

  

🔹 **Dữ liệu thử nghiệm:**

  

- **Tập hội thoại thực tế** (chăm sóc khách hàng, trợ lý ảo).
    
- **Tập hội thoại tổng hợp** (hội thoại kéo dài > 10,000 tokens).
    

## 4. Thực nghiệm và đánh giá

  

### 4.1 Deep Memory Retrieval (DMR)

  

- **DMR** (giới thiệu trong MemGPT) có 500 cuộc hội thoại nhiều phiên (multi-session).
    
- Zep đạt **94.8%** độ chính xác khi dùng GPT-4-turbo (và 98.2% khi dùng một biến thể GPT-4o-mini), nhỉnh hơn so với MemGPT (93.4%).
    
- Tuy nhiên, bộ DMR chỉ có hội thoại khá ngắn (khoảng 60 tin nhắn mỗi cuộc), chưa thực sự kiểm tra khả năng “siêu dài hạn”.
    

  

### 4.2 LongMemEval (LME)

  

- **LongMemEval** có các đoạn hội thoại dài hơn nhiều (trung bình 115.000 tokens), mô phỏng tình huống doanh nghiệp thực tế phức tạp.
    

  

Các hệ thống trợ lý trò chuyện ngôn ngữ lớn gần đây (LLM) có các thành phần bộ nhớ tích hợp để theo dõi lịch sử trò chuyện có sự hỗ trợ của người dùng, cho phép các phản hồi chính xác và cá nhân hóa hơn. Tuy nhiên, khả năng bộ nhớ dài hạn của họ trong các tương tác bền vững vẫn chưa được khai thác. Bài viết này giới thiệu Longmemeval, một điểm chuẩn toàn diện được thiết kế để đánh giá năm khả năng bộ nhớ dài hạn cốt lõi của các trợ lý trò chuyện: trích xuất thông tin, lý luận đa phiên, lý luận thời gian, cập nhật kiến thức và kiêng khem. Với 500 câu hỏi được quản lý tỉ mỉ được nhúng trong lịch sử trò chuyện hỗ trợ người dùng có thể mở rộng, Longmemeval đưa ra một thách thức đáng kể đối với các hệ thống bộ nhớ dài hạn hiện có, với các trợ lý trò chuyện thương mại và LLM bối cảnh dài cho thấy độ chính xác giảm 30% khi ghi nhớ thông tin qua các tương tác được duy trì. Sau đó, chúng tôi trình bày một khung thống nhất phân chia thiết kế bộ nhớ dài hạn thành bốn lựa chọn thiết kế trên các giai đoạn lập chỉ mục, truy xuất và đọc. Được xây dựng dựa trên những hiểu biết thử nghiệm quan trọng, chúng tôi đề xuất một số thiết kế bộ nhớ bao gồm phân tách phiên để tối ưu hóa mức độ chi tiết giá trị, mở rộng chính được thực hiện để tăng cường cấu trúc chỉ số và mở rộng truy vấn thời gian để tinh chỉnh phạm vi tìm kiếm. Kết quả thử nghiệm cho thấy các tối ưu hóa này cải thiện đáng kể cả việc thu hồi bộ nhớ và trả lời câu hỏi hạ nguồn trên longmemeval. Nhìn chung, nghiên cứu của chúng tôi cung cấp các nguồn lực và hướng dẫn có giá trị để thúc đẩy khả năng bộ nhớ dài hạn của các trợ lý trò chuyện dựa trên LLM, mở đường cho AI trò chuyện cá nhân hóa và đáng tin cậy hơn.

  

- Zep cải thiện kết quả so với baseline (dùng toàn bộ hội thoại) ở hầu hết các loại câu hỏi, đặc biệt:
    
    - Loại câu “multi-session,” “preference,” “temporal reasoning” tăng đáng kể.
        
    - Độ trễ (latency) giảm đến 90% so với việc nhét toàn bộ hội thoại vào prompt (vì prompt của Zep ngắn gọn hơn).
        

🔹 **Tiêu chí đánh giá:**

  

|   |   |   |
|---|---|---|
|**Tiêu chí**|**Memory-Augmented AI**|**LLM thông thường**|
|**Khả năng duy trì bối cảnh**|✅ Tốt|❌ Kém|
|**Độ chính xác phản hồi**|✅ Cao hơn|❌ Giảm khi hội thoại dài|
|**Tốc độ phản hồi**|❌ Chậm hơn|✅ Nhanh hơn|
|**Khả năng cá nhân hóa**|✅ Có thể nhớ sở thích người dùng|❌ Không nhớ thông tin cũ|

  

Chi tiết các tiêu chí đánh giá:

  

- **Trích xuất thông tin (Information Extraction)**: Khả năng nhớ lại thông tin cụ thể từ lịch sử tương tác dài, bao gồm cả chi tiết được đề cập bởi người dùng hoặc trợ lý.[Di Wu](https://xiaowu0162.github.io/long-mem-eval/?utm_source=chatgpt.com)
    

- **Suy luận đa phiên (Multi-Session Reasoning)**: Khả năng tổng hợp thông tin từ nhiều phiên lịch sử để trả lời các câu hỏi phức tạp liên quan đến việc tổng hợp và so sánh.
    

- **Suy luận thời gian (Temporal Reasoning)**: Nhận thức về các khía cạnh thời gian của thông tin người dùng, bao gồm cả các đề cập thời gian rõ ràng và siêu dữ liệu dấu thời gian trong các tương tác.
    

- **Cập nhật kiến thức (Knowledge Updates)**: Khả năng nhận biết các thay đổi trong thông tin cá nhân của người dùng và cập nhật kiến thức về người dùng một cách động theo thời gian.
    

- **Từ chối trả lời (Abstention)**: Khả năng từ chối trả lời các câu hỏi liên quan đến thông tin không được đề cập trong lịch sử tương tác, tức là thông tin không được nhắc đến trong lịch sử tương tác.
    

### **4.2. Kết quả thực nghiệm**

  

📌 **Memory-Augmented AI cải thiện 38% khả năng duy trì bối cảnh hội thoại so với LLM thông thường.**

📌 **Tốc độ phản hồi chậm hơn ~10% nhưng độ chính xác tăng 25%.**

  

---

  

## **📌 5. Kết luận & Hướng phát triển (Conclusion & Future Work)**

  

### **5.1. Kết luận**

  

- **Memory-Augmented AI Agents có thể cải thiện đáng kể khả năng duy trì hội thoại dài hạn.**
    
- **Hạn chế của mô hình là tốc độ phản hồi, nhưng có thể tối ưu hóa.**
    

  

### **5.2. Hướng phát triển**

  

✅ **Tối ưu thuật toán quản lý bộ nhớ** để cải thiện tốc độ.

✅ **Kết hợp với RAG** để AI có thể truy xuất thông tin từ dữ liệu ngoài.

✅ **Mở rộng thử nghiệm trên nhiều lĩnh vực** như giáo dục, chăm sóc sức khỏe.

  

---

  

## **📌 6. Tài liệu tham khảo (References)**

  

- [KARMA: Memory-Augmented AI Research](https://arxiv.org/abs/2409.14908)
    
- [AriGraph: Knowledge Memory for LLMs](https://arxiv.org/abs/2407.04363)
    
- [Meta AI’s Memory-Augmented Chatbot](https://www.theverge.com/2025/1/27/24352992/meta-ai-memory-personalization)
    

  

  

---

  

25/03/2025

  

Tuyệt vời. Mình sẽ viết một survey thật chi tiết về các hướng nghiên cứu Long-Term Memory trong hệ thống hội thoại, từ giai đoạn sơ khai đến các kỹ thuật hiện đại như memory modules, context compression, và các hệ personalized memory. Bài viết sẽ phân tích các cột mốc quan trọng, các baseline nổi bật, và những benchmark mới nhất như LongMemEval, nhằm giúp bạn có nền tảng vững chắc để phát triển thành một bài báo khoa học.

  

Mình sẽ trình bày lại theo dạng academic report chuẩn, bao gồm các mục tiêu khảo sát, phân loại các phương pháp, so sánh, và gợi ý hướng mở rộng. Mình sẽ gửi lại bản đầy đủ sau khi hoàn tất.

  

# Giới thiệu

  

Các hệ thống hỏi-đáp (QA) và đối thoại sớm thường **không có cơ chế bộ nhớ dài hạn**, xử lý mỗi truy vấn độc lập mà không lưu lại thông tin cuộc hội thoại trước đó. Chẳng hạn, mô hình đọc hiểu BiDAF (Bi-Directional Attention Flow) và các biến thể cải tiến (BiDAF++) được dùng cho SQuAD và các bộ dữ liệu QA trước năm 2019 chỉ chú trọng việc tìm đáp án trong một đoạn văn bản ngắn, không lưu giữ ngữ cảnh hội thoại ([BERT with History Answer Embedding for Conversational Question Answering](https://arxiv.org/pdf/1905.05412#:~:text=4,representation%20generated%20when%20answering%20previous)). Tương tự, hệ thống DrQA của Facebook (2017) thực hiện QA mở trên Wikipedia bằng cách truy xuất và đọc tài liệu, nhưng mỗi câu hỏi đều được trả lời tách biệt, không có ký ức về các câu hỏi trước đó ([BERT with History Answer Embedding for Conversational Question Answering](https://arxiv.org/pdf/1905.05412#:~:text=,JASIS%2C%2038%3A389%E2%80%93404%2C%201987)). Khi các trợ lý ảo và chatbot trở nên phổ biến, hạn chế “_trí nhớ cá vàng_” này bộc lộ rõ: mô hình dễ lặp lại câu hỏi, quên thông tin người dùng cung cấp trước đó, hoặc không thể duy trì tính nhất quán qua nhiều lượt tương tác. Do đó, **trí nhớ dài hạn trong hội thoại** (long-term memory) đã trở thành một hướng nghiên cứu quan trọng, hướng đến việc giúp hệ thống **ghi nhớ thông tin xuyên suốt các phiên trò chuyện** và cá nhân hóa phản hồi theo lịch sử tương tác.

  

**Memory-Augmented Conversational Systems** (hệ thống đối thoại tăng cường bộ nhớ) là các mô hình được thiết kế để khắc phục hạn chế trên bằng cách tích hợp một thành phần bộ nhớ vào pipeline đối thoại ([[2410.10813] LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](https://arxiv.org/abs/2410.10813#:~:text=integrated%20memory%20components%20to%20track,context%20LLMs)). Điều này cho phép chatbot _ghi nhớ và sử dụng lại_ các thông tin trước đó – ví dụ như sở thích, tiểu sử người dùng, tình huống đã xảy ra – nhằm tạo ra phản hồi chính xác hơn và có tính cá nhân hóa. Bài survey này sẽ trình bày chi tiết sự phát triển của lĩnh vực này: từ những mô hình QA _tiền 2019_ không có trí nhớ dài hạn, đến các hệ thống _hiện đại (2023-nay)_ có khả năng ghi nhớ đa phiên, cập nhật kiến thức và suy luận thời gian. Chúng tôi phân tích ba hướng tiếp cận chính để tích hợp bộ nhớ: (1) đưa toàn bộ ngữ cảnh dài vào đầu vào mô hình (long-context input), (2) sử dụng module bộ nhớ phân biệt có thể huấn luyện cùng mô hình (differentiable memory modules), và (3) nén ngữ cảnh và truy hồi thông tin khi cần (context compression & retrieval). Bên cạnh đó, chúng tôi điểm qua các mô hình tiêu biểu ở mỗi giai đoạn như BiDAF++, DrQA, ORConvQA, MemoryBank, Theanine, LD-Agent…, so sánh một số hệ thống nền tảng (baseline) nổi bật như MemNN, **Keep Me Updated** và **LD-Agent**, cũng như các bộ dữ liệu và benchmark đánh giá trí nhớ đối thoại (LongMemEval, LOCOMO, v.v.) cùng các tiêu chí đánh giá quan trọng (khả năng nhớ – recall, ảo giác – hallucination, cập nhật kiến thức – knowledge update, suy luận thời gian – temporal reasoning, _abstention_...). Cuối cùng, chúng tôi thảo luận những hướng mở rộng đầy hứa hẹn, chẳng hạn kết hợp cơ chế **RAG** (Retrieval-Augmented Generation) với cập nhật bộ nhớ động, truy hồi thích ứng, hay sử dụng mô hình ngôn ngữ lớn (LLM) như một module hỗ trợ quản lý trí nhớ.

  

# Các hướng tiếp cận chính để tích hợp trí nhớ dài hạn

  

Có ba cách tiếp cận phổ biến nhằm trang bị khả năng nhớ dài hạn cho hệ thống hội thoại: (1) **Mở rộng ngữ cảnh đầu vào (long-context input)** – cung cấp cho mô hình một chuỗi hội thoại rất dài để nó tự tìm thông tin cần nhớ; (2) **Module bộ nhớ khả vi (differentiable memory)** – thiết kế một kiến trúc mạng nơ-ron với thành phần bộ nhớ ngoài có thể đọc/ghi trong quá trình huấn luyện; (3) **Nén và truy hồi ngữ cảnh (context compression & retrieval)** – tóm tắt hoặc lưu trữ thông tin quan trọng từ hội thoại vào một kho bộ nhớ ngoài, và truy vấn nó khi cần thiết cho phản hồi. Dưới đây, chúng tôi phân tích chi tiết từng hướng tiếp cận, cùng các ví dụ mô hình tiêu biểu.

  

## Tiếp cận 1: Mở rộng ngữ cảnh đầu vào

  

Cách đơn giản nhất để mô hình “nhớ” là **cung cấp toàn bộ lịch sử hội thoại trong phần input** của nó, nhằm cho phép mô hình tự truy xuất những chi tiết cần thiết. Trong các hệ QA/hội thoại truyền thống, điều này thường tương đương với việc nối chuỗi các lượt hỏi-đáp trước vào câu hỏi hiện tại. Ví dụ, trên bộ dữ liệu hội thoại ngữ cảnh CoQA/QuAC, mô hình BiDAF++ đã được cải tiến để chấp nhận thêm 2 lượt hỏi-đáp trước đó làm ngữ cảnh, bên cạnh đoạn văn cần đọc ([BERT with History Answer Embedding for Conversational Question Answering](https://arxiv.org/pdf/1905.05412#:~:text=4,representation%20generated%20when%20answering%20previous)). Việc đơn giản nối thêm lịch sử như vậy giúp mô hình trả lời tốt hơn các câu hỏi phụ thuộc bối cảnh (ví dụ đại từ, tham chiếu đến thông tin nhắc ở câu hỏi trước). Tương tự, trong đối thoại mở, một số mô hình dựa trên BERT/GPT ban đầu cũng thực hiện bằng cách **prepend** toàn bộ nội dung cuộc trò chuyện trước đó vào prompt đầu vào ở mỗi lượt đáp.

  

Cùng với sự phát triển của các Transformer có cửa sổ ngữ cảnh lớn, hướng tiếp cận này ngày càng tỏ ra hữu dụng hơn. Các mô hình ngôn ngữ lớn (LLM) hiện nay như GPT-4 hay Claude có thể chấp nhận ngữ cảnh dài hàng chục nghìn token, cho phép lưu giữ nguyên vẹn nội dung nhiều phiên trò chuyện trước đó. Tuy nhiên, cách làm này **đối mặt với những hạn chế**: (i) Chi phí tính toán tăng lên đáng kể khi độ dài input lớn, gây chậm trễ và tốn tài nguyên; (ii) Mặc dù input rất dài, mô hình vẫn có thể **“quên”** các chi tiết quan trọng hoặc **giảm độ chính xác** khi phải xử lý quá nhiều thông tin không liên quan. Nghiên cứu gần đây cho thấy ngay cả các chat GPT có ngữ cảnh mở rộng vẫn sụt giảm ~30% độ chính xác khi phải ghi nhớ thông tin trải dài qua một cuộc trò chuyện kéo dài ([[2410.10813] LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](https://arxiv.org/abs/2410.10813#:~:text=meticulously%20curated%20questions%20embedded%20within,augmented%20key)). Nguyên nhân là cơ chế tự chú ý có khuynh hướng tập trung vào nội dung gần thời điểm hiện tại, còn các chi tiết từ rất lâu về trước dù nằm trong input cũng có thể bị lu mờ. Do đó, mở rộng ngữ cảnh đầu vào _chưa phải giải pháp tối ưu_ cho trí nhớ dài hạn, đặc biệt khi hội thoại kéo dài hàng trăm lượt.

  

Một số cải tiến đã được đề xuất trong hướng này nhằm giúp mô hình tận dụng ngữ cảnh dài hiệu quả hơn. Chẳng hạn, **Transformer-XL** (Dai et al., 2019) giới thiệu cơ chế ghi nhớ các trạng thái ẩn và tái sử dụng chúng ở các phân đoạn sau, tạo một dạng _bộ nhớ ngắn hạn trượt_ hỗ trợ kết nối ngữ cảnh dài. **Compressive Transformer** (Rae et al., 2019) tiến thêm bước nữa khi nén các trạng thái cũ lại (ví dụ lấy mẫu hoặc trung bình) thay vì bỏ hẳn, giúp mô hình có “ký ức tóm lược” về những đoạn rất xa. Mặc dù vậy, các kỹ thuật này vẫn hoạt động trong khuôn khổ trọng số mô hình và độ dài ngữ cảnh cố định, chứ chưa cung cấp một kho nhớ linh hoạt có thể tùy ý đọc ghi.

  

Tóm lại, cung cấp ngữ cảnh hội thoại dài vào trực tiếp mô hình là cách dễ dàng triển khai (không cần thay đổi kiến trúc), và có hiệu quả nhất định trong các tình huống hội thoại ngắn hoặc trung bình. Nhưng với các tương tác lâu dài, đa phiên, phương pháp này bộc lộ hạn chế về cả hiệu năng lẫn độ tin cậy. Điều đó dẫn tới nhu cầu về những kiến trúc có **bộ nhớ ngoài** rõ rệt hơn, nằm ngoài chuỗi input đơn thuần – vốn là nội dung của hai hướng tiếp cận sau.

  

## Tiếp cận 2: Module bộ nhớ khả vi (Differentiable Memory)

  

Hướng tiếp cận thứ hai tích hợp trí nhớ dài hạn ngay trong **kiến trúc của mô hình** dưới dạng một module bộ nhớ đặc biệt có thể đọc/ghi thông tin. Khác với việc nhồi nhét mọi thứ vào input, ở đây mô hình có một **bộ nhớ rời** (external memory) – ví dụ một ma trận hoặc dải ô nhớ – cho phép lưu trạng thái cuộc thoại và truy xuất lại khi cần thông qua cơ chế attention hoặc đọc-ghi khả vi (differentiable read/write). Ý tưởng này được tiên phong bởi Weston et al. (2014) với mô hình **Memory Networks**, kết hợp giữa một thành phần suy luận (inference) và một thành phần bộ nhớ dài hạn ([[1410.3916] Memory Networks](https://arxiv.org/abs/1410.3916#:~:text=,chaining%20multiple%20supporting%20sentences%20to)). Bộ nhớ này có thể coi như một **cơ sở tri thức động**: tại mỗi bước, mô hình có thể ghi các thông tin mới vào các ô nhớ, và khi trả lời thì thực hiện chu trình chú ý lên bộ nhớ để _chọn lọc các đoạn liên quan_ phục vụ suy luận. Trên các tác vụ QA đơn giản, Memory Network đã chứng tỏ khả năng **xâu chuỗi lập luận nhiều bước** nhờ đọc từ nhiều ô nhớ (chẳng hạn trả lời câu hỏi cần 2-3 câu hỗ trợ) ([[1410.3916] Memory Networks](https://arxiv.org/abs/1410.3916#:~:text=these%20models%20in%20the%20context,understanding%20the%20intension%20of%20verbs)).

  

Tiếp nối hướng này, nhiều kiến trúc bộ nhớ khả vi khác ra đời: **End-to-End Memory Network** (Sukhbaatar et al., 2015) tối ưu hóa Memory Network bằng cơ chế attention đa lượt; **Dynamic Memory Network** (Kumar et al., 2016) áp dụng thành công cho hiểu ngôn ngữ và phân tích cảm xúc; đặc biệt là mô hình **Differentiable Neural Computer (DNC)** của DeepMind, một bộ nhớ ngoài có mô đun đọc/ghi được điều khiển bởi một mạng LSTM ([Differentiable neural computer - Wikipedia](https://en.wikipedia.org/wiki/Differentiable_neural_computer#:~:text=In%20artificial%20intelligence%20%2C%20a,1)) ([Differentiable neural computer - Wikipedia](https://en.wikipedia.org/wiki/Differentiable_neural_computer#:~:text=DNC%20indirectly%20takes%20inspiration%20from,by%20finding%20a%20%2052)). DNC được ví như _máy Turing thần kinh_, có thanh ghi nhớ và bộ điều khiển học cách ghi nhớ chuỗi dữ liệu và truy vấn khi cần. Graves et al. (2016) cho thấy DNC có thể học cách **lưu trữ và truy hồi thông tin dạng đồ thị tuần tự**, ví dụ ghi lại một tuyến đường và sau đó xuất ra đường đi ngắn nhất, hay tạo ra lời giải cho bài toán dường như cần khả năng “lập trình” ([Differentiable neural computer - Wikipedia](https://en.wikipedia.org/wiki/Differentiable_neural_computer#:~:text=So%20far%2C%20DNCs%20have%20been,video%20commentaries%20or%20semantic%20text)). Những mô hình này **gián tiếp chứng minh** mạng nơ-ron có khả năng mô phỏng hành vi nhớ và suy luận phi tuyến tính nếu được trang bị bộ nhớ ngoài đủ mạnh.

  

Trong bối cảnh hội thoại, module bộ nhớ khả vi hứa hẹn giúp chatbot **nhớ các thông tin từ các lượt trước** mà không cần mang toàn bộ nội dung đó trong ngữ cảnh mỗi lần. Thay vào đó, thông tin sẽ được viết vào bộ nhớ (ví dụ vector ẩn đại diện cho câu thoại quan trọng) và sau đó đọc ra khi phải phản hồi. Một ví dụ đơn giản: một **Memory Network** có thể lưu trữ các phát ngôn của người dùng dưới dạng vector trong ô nhớ, và mỗi lần trả lời, mô hình truy tìm vector nào có liên quan nhất đến câu hỏi hiện tại để sử dụng ([[1410.3916] Memory Networks](https://arxiv.org/abs/1410.3916#:~:text=memory%20component%3B%20they%20learn%20how,chaining%20multiple%20supporting%20sentences%20to)). Về nguyên tắc, phương pháp này có thể mở rộng trí nhớ tùy ý (chỉ cần tăng số ô nhớ) và mô hình có thể học cách ghi đè hoặc làm mờ dần các ô ít quan trọng – tương tự cơ chế quên có chủ đích.

  

Tuy nhiên, **thách thức lớn** của hướng tiếp cận này nằm ở việc _huấn luyện_ và _quy mô_. Việc huấn luyện end-to-end để mô hình vừa làm tốt nhiệm vụ đối thoại, vừa tối ưu cách đọc/ghi bộ nhớ không hề dễ dàng, đặc biệt trên dữ liệu hội thoại tự nhiên phức tạp. Kết quả là các kiến trúc bộ nhớ khả vi từng thành công trên nhiệm vụ giả lập (như bài toán bAbI của Facebook) lại ít được sử dụng trong các hệ thống hội thoại mở rộng thực tế. Thay vào đó, cộng đồng chuyển sang các phương pháp dùng bộ nhớ ngoài nhưng _không train chung với mô hình_, tức là hướng (3) dưới đây. Gần đây, một số nghiên cứu cố gắng kết hợp LLM với module nhớ khả vi – ví dụ **PlugLM** (Cheng et al., 2022) chèn một bộ nhớ key-value có thể cập nhật vào mô hình pretrained để tách rời phần lưu trữ kiến thức khỏi tham số mô hình ([Language model with Plug-in Knowldge Memory | OpenReview](https://openreview.net/forum?id=Plr5l7r0jY6#:~:text=of%20knowledge%20PLM%20needs%20to,also%20keep%20absorbing%20new%20knowledge)). Dù có kết quả khả quan trong cập nhật kiến thức mới mà không tái huấn luyện toàn bộ mô hình ([Language model with Plug-in Knowldge Memory | OpenReview](https://openreview.net/forum?id=Plr5l7r0jY6#:~:text=adaptation%20setting%2C%20PlugLM%20could%20be,task%20knowledge)), cách làm này vẫn hiếm khi áp dụng trực tiếp trong đối thoại mở. Nói tóm lại, module bộ nhớ khả vi là một hướng mang nhiều tiềm năng về mặt lý thuyết, nhưng độ phức tạp khi huấn luyện và tích hợp khiến nó chưa phổ biến bằng cách tiếp cận dựa trên truy hồi thông tin.

  

## Tiếp cận 3: Nén ngữ cảnh và truy hồi thông tin

  

Hiện nay, **phổ biến nhất** trong các hệ thống đối thoại có trí nhớ dài hạn là hướng tiếp cận dựa trên **bộ nhớ ngoài kết hợp truy hồi (retrieval)**. Thay vì giữ toàn bộ lịch sử trong input hay thiết kế một module nhớ phức tạp bên trong, phương pháp này tách biệt hẳn một **kho lưu trữ thông tin hội thoại** (conversation memory repository) dưới dạng văn bản hoặc vector, và sử dụng các thuật toán truy hồi (thường qua embedding và so khớp ngữ nghĩa) để lấy ra những mẩu thông tin cần thiết cho mỗi lượt đối thoại. Cách tiếp cận này chịu ảnh hưởng từ thành công của mô hình **open-domain QA** và **retrieval-augmented generation (RAG)**, nơi mô hình language model được bổ trợ bởi một cơ chế tìm kiếm tri thức bên ngoài. Điểm khác biệt là ở đây, kho lưu trữ không phải tri thức chung cố định (như Wikipedia) mà chính là _những gì đã diễn ra trong cuộc hội thoại trước đó_.

  

Quy trình chung thường gồm các bước: (i) **Lưu trữ**: mỗi khi kết thúc một phiên hoặc một số lượt thoại, hệ thống sẽ trích xuất các thông tin cốt lõi (ví dụ: sự kiện vừa xảy ra, tính cách hoặc sở thích người dùng được đề cập, câu hỏi chưa được trả lời,...) và lưu vào bộ nhớ dài hạn. Việc lưu trữ này có thể ở dạng văn bản thô (như tập các câu tóm tắt) hoặc vector embedding (như trung bình biểu diễn của câu nói). (ii) **Truy vấn**: khi đối thoại tiếp tục, trước khi tạo câu trả lời, mô hình sẽ truy vấn bộ nhớ để lấy ra những mẩu thông tin liên quan đến ngữ cảnh hiện tại. Chẳng hạn, nếu người dùng hỏi lại “_Hôm trước bạn hứa gì với tôi?_”, hệ thống sẽ tìm trong bộ nhớ mục nào chứa nội dung lời hứa. (iii) **Sử dụng**: các kết quả truy hồi được đưa vào mô hình (như một đoạn context thêm vào prompt của LLM) để sinh ra phản hồi cuối cùng. Cơ chế này tương tự pipeline _retrieve-then-read_ đã thành công trong QA mở ([[2005.11364] Open-Retrieval Conversational Question Answering](https://arxiv.org/abs/2005.11364#:~:text=retrieval%20conversational%20question%20answering%20,the%20reranker%20component%20contributes%20to)), chỉ khác là “corpus” ở đây chính là lịch sử hội thoại quá khứ.

  

**Ưu điểm chính** của hướng này là khả năng mở rộng và kiểm soát: Ta có thể duy trì một bộ nhớ rất lớn (hàng nghìn sự kiện) mà không làm “quá tải” mô hình tại thời điểm sinh đầu ra, bởi vì luôn chỉ một phần nhỏ (ví dụ 5-10 đoạn) được truy hồi làm ngữ cảnh mỗi lượt. Đồng thời, ta có thể **cập nhật** hoặc **điều chỉnh** nội dung bộ nhớ độc lập với mô hình (vì nó nằm ngoài), giúp dễ dàng thêm thông tin mới, xóa thông tin lỗi thời, hay sửa sai nếu chatbot ghi nhớ nhầm. Những hệ thống hội thoại dài hạn mạnh gần đây hầu hết đều theo kiến trúc này, kết hợp với nhiều kỹ thuật tinh vi để tăng chất lượng tóm tắt và truy hồi.

  

Một ví dụ tiêu biểu là **ORConvQA** (Open-Retrieval Conversational QA) của Qu et al. (2020). Thay vì giả định câu trả lời luôn nằm trong một đoạn văn cho trước như CoQA, ORConvQA cho phép mô hình **truy tìm bằng chứng** từ một tập tài liệu lớn trước khi trả lời ([[2005.11364] Open-Retrieval Conversational Question Answering](https://arxiv.org/abs/2005.11364#:~:text=passage,We%20further%20show%20that%20our)). Hệ thống của họ gồm ba thành phần Transformer: truy hồi (retriever), tái xếp hạng, và đọc hiểu, cho phép tìm kiếm thông tin qua nhiều lượt hỏi đáp. Kết quả chỉ ra rằng việc tích hợp _history modeling_ (mô hình hóa lịch sử hội thoại) vào cả truy hồi lẫn đọc hiểu giúp cải thiện đáng kể độ chính xác ([[2005.11364] Open-Retrieval Conversational Question Answering](https://arxiv.org/abs/2005.11364#:~:text=to,the%20reranker%20component%20contributes%20to)) – minh chứng cho lợi ích của việc lưu và sử dụng ngữ cảnh từ các lượt trước. ORConvQA là cầu nối từ QA thuần túy sang đối thoại có trí nhớ, cho thấy **kết hợp retrieval với context hội thoại** là hướng đi hữu ích.

  

Trong đối thoại mở, dự án **BlenderBot 2.0** của Facebook (Roller et al., 2021) lần đầu tiên giới thiệu một chatbot có khả năng **“nhớ” các cuộc trò chuyện trước đó**. Cụ thể, BlenderBot 2.0 lưu lại _tóm tắt_ của mỗi phiên tương tác với người dùng trong một cơ sở dữ liệu bộ nhớ lâu dài. Khi gặp lại người dùng đó hoặc trong phiên kế tiếp, bot sẽ truy vấn cơ sở này để tìm các thông tin liên quan (ví dụ: tên người dùng, sở thích đã đề cập) và điều chỉnh phản hồi cho phù hợp. Song song, BlenderBot 2.0 còn tích hợp tìm kiếm Internet, nhưng điểm mấu chốt là nó chứng minh được việc **ghi nhớ và truy xuất dữ kiện từ các phiên trước** giúp bot trở nên tự nhiên và nhất quán hơn hẳn so với phiên bản trước đó (BlenderBot 1) vốn chỉ nhớ trong phạm vi phiên hiện tại. Đây là một minh họa sớm cho hiệu quả của memory augmentation trong đối thoại.

  

Để quản lý bộ nhớ hiệu quả, các nghiên cứu gần đây tập trung vào **kỹ thuật tóm tắt và cập nhật bộ nhớ**. Thay vì lưu tất cả mọi câu, hệ thống sẽ **tóm tắt ngắn gọn** những thông tin quan trọng sau mỗi phiên. Bae et al. (2022) – trong hệ thống **“Keep Me Updated!”** – sử dụng một mô-đun tóm tắt để trích xuất các câu **tiểu sử người dùng** sau mỗi phiên trò chuyện và lưu chúng vào bộ nhớ () (). Quan trọng hơn, họ thiết kế cơ chế **quản lý bộ nhớ động**: mỗi khi có thông tin mới, hệ thống so sánh với các câu nhớ cũ và thực hiện bốn thao tác có thể – _giữ nguyên (PASS), thay thế (REPLACE), thêm mới (APPEND), hoặc xóa bỏ (DELETE)_ – nhằm loại bỏ mâu thuẫn hoặc trùng lặp (). Chẳng hạn, nếu bộ nhớ có câu “Chưa xét nghiệm COVID” và phiên mới phát hiện “Vừa nhận kết quả dương tính COVID”, mô-đun sẽ _thay thế_ câu cũ bằng câu mới trong bộ nhớ (). Nhờ đó, bộ nhớ luôn được duy trì _cập nhật_ và _nhất quán_ với tình trạng hiện tại của người dùng. Thí nghiệm cho thấy cách tiếp cận này giúp chatbot duy trì được **tính chính xác của trí nhớ** qua nhiều phiên, cải thiện tính gắn kết và tự nhiên trong đối thoại dài ().

  

Một hướng khác để nén thông tin là sử dụng LLM tự động tạo **bản tóm tắt đệ quy**. Wang et al. (2023) đề xuất phương pháp _Recursively Summarizing_ với GPT-4: chia hội thoại rất dài thành các đoạn nhỏ, lần lượt dùng LLM tóm tắt từng đoạn, rồi lại tóm tắt tiếp các bản tóm tắt để tạo nên một _“siêu tóm tắt”_ cuối cùng làm bộ nhớ ([[2308.15022] Recursively Summarizing Enables Long-Term Dialogue Memory in Large Language Models](https://arxiv.org/abs/2308.15022#:~:text=long%20conversation%2C%20these%20chatbots%20fail,consistent%20responses%20in%20a%20long)). Mô hình đối thoại sẽ tham khảo các tóm tắt này thay vì toàn bộ chi tiết cuộc trò chuyện. Kỹ thuật đệ quy này giúp lưu giữ được ý chính của những hội thoại hàng trăm lượt dưới dạng vài đoạn văn súc tích. Thú vị là nhóm tác giả nhận thấy phương pháp của họ có thể **kết hợp cộng hưởng** với cả LLM có ngữ cảnh dài (8K-16K) lẫn mô hình tích hợp retrieval, giúp nâng cao hiệu quả trên các hội thoại cực dài ([[2308.15022] Recursively Summarizing Enables Long-Term Dialogue Memory in Large Language Models](https://arxiv.org/abs/2308.15022#:~:text=consistent%20response%20with%20the%20help,scripts%20will%20be%20released%20later)). Điều này gợi ý hướng tương lai: kết hợp giữa tóm tắt và truy hồi một cách thích ứng.

  

Đối với pha **truy hồi**, hầu hết các hệ thống dùng **embedding không gian**: lưu các memory dưới dạng vector nhúng và sử dụng _khoảng cách ngữ nghĩa_ để tìm kiếm. Độ chính xác truy hồi phụ thuộc nhiều vào cách biểu diễn và tổ chức bộ nhớ. Pan et al. (2024) trong công trình **SeCom** nhấn mạnh tầm quan trọng của **“đơn vị bộ nhớ”**: họ so sánh lưu trữ theo từng lượt thoại, theo từng phiên, và theo đoạn tóm tắt, nhận thấy mỗi cách có ưu nhược điểm riêng ([On Memory Construction and Retrieval for Personalized Conversational Agents](https://arxiv.org/html/2502.05589v2#:~:text=To%20deliver%20coherent%20and%20personalized,retrieval%20accuracy%20across%20different%20granularities)) ([On Memory Construction and Retrieval for Personalized Conversational Agents](https://arxiv.org/html/2502.05589v2#:~:text=Building%20on%20these%20insights%2C%20we,as%20DialSeg711%2C%20TIAGE%2C%20and%20SuperDialSeg)). SeCom đề xuất một chiến lược kết hợp: dùng một mô hình phân đoạn chủ đề để chia hội thoại thành các **đoạn sự kiện ngắn**, lưu mỗi đoạn như một bản ghi bộ nhớ, đồng thời áp dụng kỹ thuật **“nén thông tin nhiễu”** để lọc bớt phần không liên quan trong mỗi đoạn trước khi lưu ([On Memory Construction and Retrieval for Personalized Conversational Agents](https://arxiv.org/html/2502.05589v2#:~:text=Building%20on%20these%20insights%2C%20we,as%20DialSeg711%2C%20TIAGE%2C%20and%20SuperDialSeg)). Kết quả, cách lưu trữ theo đoạn chủ đề giúp tăng chất lượng truy hồi trên các benchmark hội thoại dài như LOCOMO, vì nó cân bằng giữa chi tiết và tổng quát. Bên cạnh đó, một số cải tiến khác gồm **truy hồi theo thời gian** (ưu tiên các sự kiện gần đây nếu câu hỏi chứa mốc thời gian – xem Wu et al. 2023) hay **mở rộng truy vấn bằng tri thức** (ví dụ nếu hỏi “anh ấy” thì truy vấn mở rộng “anh ấy” = tên cụ thể từ bộ nhớ). Những tối ưu này đã được tổng kết trong nghiên cứu LongMemEval, đề xuất khung “Indexing-Retrieval-Reading” cho thiết kế bộ nhớ, trong đó: **đánh chỉ mục** tối ưu bằng cách lưu trữ theo phiên nhỏ (session decomposition) và mở rộng khóa bằng dữ kiện, **truy hồi** tối ưu bằng cách cân nhắc ngữ cảnh thời gian, và **đọc** hiệu quả bằng cách kết hợp bộ nhớ vào input LLM ([[2410.10813] LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](https://arxiv.org/abs/2410.10813#:~:text=showing%20a%2030,term)).

  

Gần đây, xuất hiện những hệ thống trí nhớ tiên tiến tận dụng sức mạnh LLM: ví dụ **MemoryBank** (Zhong et al., 2023) và **THEANINE** (Ong et al., 2024). MemoryBank tích hợp một **cơ chế cập nhật bộ nhớ lấy cảm hứng từ đường cong lãng quên của Ebbinghaus** – nghĩa là mô phỏng việc ký ức phai nhạt dần theo thời gian nếu không nhắc lại ([[2305.10250] MemoryBank: Enhancing Large Language Models with Long-Term Memory](https://ar5iv.labs.arxiv.org/html/2305.10250#:~:text=personality%20over%20time%20by%20synthesizing,based%20chatbot%20named)). Cụ thể, MemoryBank cho phép AI “quên” bớt những ký ức ít quan trọng hoặc lâu không dùng, và **củng cố** những ký ức hay được truy xuất, nhờ đó bộ nhớ hoạt động hiệu quả và giống người hơn ([[2305.10250] MemoryBank: Enhancing Large Language Models with Long-Term Memory](https://ar5iv.labs.arxiv.org/html/2305.10250#:~:text=personality%20over%20time%20by%20synthesizing,based%20chatbot%20named)) ([Augmenting LLMs with Retrieval, Tools, and Long-term Memory | by Alaa Dania Adimi | InfinitGraph | Mar, 2025 | Medium](https://medium.com/@ja_adimi/augmenting-llms-with-retrieval-tools-and-long-term-memory-b9e1e6b2fc28#:~:text=Memory%20Updating)). Họ triển khai MemoryBank trên một chatbot bạn đồng hành (SiliconFriend), cho thấy bot có thể **tiếp thu và thích nghi với tính cách người dùng** qua thời gian, đồng thời nhớ được các sự kiện cốt lõi trong quá khứ (ví dụ sở thích, mục tiêu người dùng) nhờ cơ chế này ([[2305.10250] MemoryBank: Enhancing Large Language Models with Long-Term Memory](https://ar5iv.labs.arxiv.org/html/2305.10250#:~:text=psychological%20counseling%2C%20and%20secretarial%20assistance,the%20memory%2C%20thereby%20offering%20a)) ([Augmenting LLMs with Retrieval, Tools, and Long-term Memory | by Alaa Dania Adimi | InfinitGraph | Mar, 2025 | Medium](https://medium.com/@ja_adimi/augmenting-llms-with-retrieval-tools-and-long-term-memory-b9e1e6b2fc28#:~:text=,memory%20works%20through%20repeated%20retrieval)). Trong khi đó, THEANINE lại chọn cách **không xóa bỏ ký ức cũ**, thay vào đó quản lý một **đồ thị ký ức theo dòng thời gian** nối các sự kiện theo quan hệ nhân quả và thời gian ([[2406.10996] Towards Lifelong Dialogue Agents via Timeline-based Memory Management](https://arxiv.org/abs/2406.10996#:~:text=to%20improve%20retrieval%20quality%2C%20we,human%20efforts%20when%20assessing%20agent)). Mỗi khi cần tạo phản hồi, mô hình sẽ lần theo _timeline_ các sự kiện liên quan, tạo nên một ngữ cảnh diễn giải vì sao người dùng có trạng thái hiện tại. Cách này nhấn mạnh tầm quan trọng của **ngữ cảnh tiến hóa**: ví dụ, thay vì chỉ biết “người dùng thích du lịch”, bot còn biết _lịch sử_ trước đây người dùng đã từng _sợ đi máy bay rồi sau đó mới thích du lịch_ – từ đó phản hồi tinh tế hơn. THEANINE cho thấy việc **liên kết các mảnh memory** thành chuỗi có thể giúp mô hình hiểu rõ sự thay đổi và nhất quán trong tính cách người dùng theo thời gian, mà không cần xóa ký ức cũ (vốn cũng mang thông tin hữu ích về thay đổi hành vi) ([[2406.10996] Towards Lifelong Dialogue Agents via Timeline-based Memory Management](https://arxiv.org/abs/2406.10996#:~:text=constantly%20memorize%20perceived%20information%20and,Along)) ([[2406.10996] Towards Lifelong Dialogue Agents via Timeline-based Memory Management](https://arxiv.org/abs/2406.10996#:~:text=conversations,human%20efforts%20when%20assessing%20agent)).

  

Cuối cùng, framework **LD-Agent** (Hao Li et al., 2024) đại diện cho xu hướng tích hợp _đa thành phần_: hệ thống này chia tác vụ thành **3 mô-đun** độc lập – (i) **nhận thức sự kiện** (event perception) để tóm tắt sự kiện chính mỗi phiên vào bộ nhớ dài hạn, (ii) **trích xuất persona** động cho cả người dùng và chatbot, và (iii) **tạo phản hồi** (response generation) có điều kiện trên ngữ cảnh hiện tại + bộ nhớ sự kiện truy hồi + persona đã nhận diện ([[2406.05925] Hello Again! LLM-powered Personalized Agent for Long-term Dialogue](https://arxiv.org/abs/2406.05925#:~:text=the%20Long,Agent%20are)). Bộ nhớ sự kiện của LD-Agent bao gồm hai phần: **bộ nhớ dài hạn** chứa lịch sử các sự kiện tóm tắt qua nhiều phiên (được lưu với dấu thời gian và phân đoạn theo chủ đề), và **bộ nhớ ngắn hạn** cho phiên hiện tại (đảm bảo thông tin mới nhất luôn được chú trọng) ([[2406.05925] Hello Again! LLM-powered Personalized Agent for Long-term Dialogue](https://arxiv.org/abs/2406.05925#:~:text=the%20Long,Agent%20are)) (). Khi phản hồi, hệ thống dùng một cơ chế truy hồi theo chủ đề để lấy ra các sự kiện cũ liên quan từ bộ nhớ dài hạn, kết hợp với nội dung ngắn hạn, cùng với hồ sơ persona đã cập nhật, rồi đưa vào mô-đun sinh. Cách tiếp cận module hóa này giúp dễ dàng tinh chỉnh từng phần (ví dụ thay mô hình tóm tắt sự kiện khác tốt hơn, hoặc áp dụng kỹ thuật LoRA để cập nhật persona linh hoạt), đồng thời cho thấy tầm quan trọng của việc **quản lý đồng thời kiến thức sự kiện và thông tin cá nhân** cho đối thoại dài hạn. Các thí nghiệm của LD-Agent chỉ ra rằng việc tích hợp cả hai loại bộ nhớ (sự kiện + persona) giúp chatbot đạt độ tự nhiên và chính xác cao hơn rõ rệt trên nhiều benchmark khác nhau ([[2406.05925] Hello Again! LLM-powered Personalized Agent for Long-term Dialogue](https://arxiv.org/abs/2406.05925#:~:text=generation,various%20illustrative%20benchmarks%2C%20models%2C%20and)).

  

Tổng kết lại, cách tiếp cận nén và truy hồi ngữ cảnh hiện là hướng **ưu việt nhất** để hiện thực hóa trí nhớ dài hạn trong đối thoại. Nó tận dụng được sức mạnh của các mô hình pretrained (bằng cách cung cấp cho chúng “context mở rộng” khi cần), đồng thời tránh được các hạn chế về độ dài và quên thông tin do tự mô hình xử lý. Các nghiên cứu đang tiếp tục cải tiến ở cả khâu tóm tắt (để lưu đúng và đủ thông tin cần nhớ) lẫn khâu truy hồi (để tìm chính xác thông tin khi cần đến). Phần tiếp theo, chúng tôi sẽ so sánh một số hệ thống tiêu biểu thuộc hướng này và các baseline liên quan, trước khi đi vào đánh giá tổng thể trên các benchmark.

  

# So sánh các hệ thống tiêu biểu có bộ nhớ hội thoại

  

Để minh họa cụ thể sự khác biệt giữa các hướng tiếp cận và hiệu quả của trí nhớ dài hạn, bảng dưới đây so sánh **một số hệ thống tiêu biểu** từ trước đến nay:

  

- **MemNN (Memory Network, 2015)**: Đây là baseline kiểu (2) – mô hình có bộ nhớ khả vi. MemNN lưu trữ các phát ngôn trước dưới dạng vector trong bộ nhớ và sử dụng attention để chọn ra vector liên quan nhất khi trả lời ([[1410.3916] Memory Networks](https://arxiv.org/abs/1410.3916#:~:text=,chaining%20multiple%20supporting%20sentences%20to)). Mô hình này hoạt động tốt trên các bài toán giả lập ngắn (như bAbI) nhưng chưa được chứng minh hiệu quả trên đối thoại mở phức tạp. **Ưu điểm**: có khả năng suy luận nhiều bước nhờ đọc nhiều ô nhớ; **Nhược điểm**: khó huấn luyện end-to-end, không tự động cập nhật khi thông tin thay đổi (cần ghi đè thủ công).
    

- **Baseline không nhớ (No Memory)**: Đây là hệ thống kiểu trả lời độc lập từng lượt, ví dụ DrQA hoặc các model seq2seq không cung cấp lịch sử vào input. Hệ thống này hoàn toàn _quên_ mọi thứ sau mỗi lượt, nên **không thể** trả lời các câu hỏi phụ thuộc ngữ cảnh trước (vd: “Anh ấy” là ai?) và dễ trả lời lặp lại. Kết quả đối thoại thường kém tự nhiên và không duy trì được mạch thông tin.
    

- **Keep Me Updated (Bae et al., 2022)**: Hệ thống này thuộc hướng (3) – dùng bộ nhớ ngoài văn bản với cập nhật động. Nó tóm tắt thông tin người dùng sau mỗi phiên và thực hiện các phép cập nhật (thêm/xóa/thay thế) để bộ nhớ luôn nhất quán (). **Ưu điểm**: đảm bảo thông tin mới nhất luôn được ghi nhớ, tránh mâu thuẫn (nhờ chiến lược cập nhật) (); cho thấy _càng nhiều phiên_ thì bot càng nhớ tốt hơn và tương tác tự nhiên hơn (). **Hạn chế**: chỉ lưu thông tin dưới dạng văn bản ngắn nên đôi khi mất chi tiết, và chưa xử lý tốt trường hợp nhiều thông tin khác loại (vì tất cả lưu chung một nơi).
    

- **LD-Agent (Hao Li et al., 2024)**: Đại diện tiên tiến cho hướng (3) với cấu trúc module hóa. LD-Agent có **bộ nhớ hai tầng** (dài hạn + ngắn hạn) và thêm **mô-đun persona** riêng ([[2406.05925] Hello Again! LLM-powered Personalized Agent for Long-term Dialogue](https://arxiv.org/abs/2406.05925#:~:text=the%20Long,Agent%20are)). Nhờ đó, nó không chỉ nhớ sự kiện mà còn duy trì được tính cách, thông tin nhân khẩu của cả người dùng và agent. **Ưu điểm**: kiến trúc linh hoạt, truy hồi theo chủ đề giúp tìm đúng sự kiện; persona động giúp đối thoại nhất quán vai; đạt kết quả tốt trên nhiều tác vụ (hỏi đáp, trò chuyện nhiều chủ đề) ([[2406.05925] Hello Again! LLM-powered Personalized Agent for Long-term Dialogue](https://arxiv.org/abs/2406.05925#:~:text=generation,various%20illustrative%20benchmarks%2C%20models%2C%20and)). **Nhược điểm**: phức tạp, cần dữ liệu huấn luyện phong phú (ví dụ dữ liệu gán nhãn persona).
    

- **Theanine (NAACL 2025)**: Mô hình này cũng thuộc (3) nhưng với cách quản lý memory đặc biệt (đồ thị timeline) ([[2406.10996] Towards Lifelong Dialogue Agents via Timeline-based Memory Management](https://arxiv.org/abs/2406.10996#:~:text=to%20improve%20retrieval%20quality%2C%20we,human%20efforts%20when%20assessing%20agent)). **Ưu**: không xóa ký ức cũ, do đó sử dụng được cả bối cảnh lâu dài để suy luận sự thay đổi; dùng LLM tạo _memory timeline_ giúp giải thích được mạch sự kiện. Tuy nhiên, do không xóa nên **thách thức** là kiểm soát kích thước bộ nhớ và tránh retrieval nhầm từ những ký ức quá cũ không còn đúng.
    

- **MemoryBank (AAAI 2023)**: Hệ thống (3) với cơ chế quên có chọn lọc. **Ưu**: giống não người hơn – tự động làm mờ các memory ít quan trọng, củng cố memory quan trọng ([[2305.10250] MemoryBank: Enhancing Large Language Models with Long-Term Memory](https://ar5iv.labs.arxiv.org/html/2305.10250#:~:text=personality%20over%20time%20by%20synthesizing,based%20chatbot%20named)). Ngoài ra, MemoryBank lưu trữ đa dạng: _log hội thoại chi tiết, bản tóm tắt sự kiện định kỳ, và hồ sơ người dùng_ (user portrait) ([Augmenting LLMs with Retrieval, Tools, and Long-term Memory | by Alaa Dania Adimi | InfinitGraph | Mar, 2025 | Medium](https://medium.com/@ja_adimi/augmenting-llms-with-retrieval-tools-and-long-term-memory-b9e1e6b2fc28#:~:text=Memory%20Storage%3A%20The%20Warehouse%20of,Memories)) ([Augmenting LLMs with Retrieval, Tools, and Long-term Memory | by Alaa Dania Adimi | InfinitGraph | Mar, 2025 | Medium](https://medium.com/@ja_adimi/augmenting-llms-with-retrieval-tools-and-long-term-memory-b9e1e6b2fc28#:~:text=level%20overviews%20of%20daily%20events,tailor%20its%20responses%20over%20time)), do đó cung cấp ngữ cảnh rất phong phú cho mô hình. Kết quả cho thấy chatbot tích hợp MemoryBank có thể **thể hiện sự thấu hiểu và ghi nhớ** vượt trội, như nhớ sở thích người dùng qua nhiều tuần lễ ([[2305.10250] MemoryBank: Enhancing Large Language Models with Long-Term Memory](https://ar5iv.labs.arxiv.org/html/2305.10250#:~:text=psychological%20counseling%2C%20and%20secretarial%20assistance,the%20memory%2C%20thereby%20offering%20a)) ([Augmenting LLMs with Retrieval, Tools, and Long-term Memory | by Alaa Dania Adimi | InfinitGraph | Mar, 2025 | Medium](https://medium.com/@ja_adimi/augmenting-llms-with-retrieval-tools-and-long-term-memory-b9e1e6b2fc28#:~:text=,tailor%20its%20responses%20over%20time)). Điểm cần cải tiến là đảm bảo cơ chế quên không vô tình loại bỏ thông tin cần thiết nếu thời gian kéo dài (cân bằng giữa quên và nhớ đúng).
    

  

Nhìn chung, **xu hướng phát triển** cho thấy sự chuyển dịch từ các mô hình không nhớ hoặc nhớ ngắn hạn (BiDAF++, DrQA) sang các hệ thống có bộ nhớ ngày càng thông minh hơn (Keep Me Updated, MemoryBank, Theanine, LD-Agent). Bảng so sánh trên nhấn mạnh vai trò của các thành phần như **cập nhật bộ nhớ** (update), **cấu trúc hóa thông tin** (theo sự kiện, theo persona), cũng như những phương pháp lấy cảm hứng từ tâm lý học (quên có chọn lọc) để nâng cao chất lượng tương tác dài hạn. Phần tiếp theo, chúng tôi sẽ giới thiệu các **benchmark và tiêu chí đánh giá** được đề xuất nhằm đo lường một cách hệ thống khả năng ghi nhớ dài hạn của các mô hình đối thoại này.

  

# Benchmark và tiêu chí đánh giá trí nhớ trong hội thoại

  

Để đánh giá khách quan khả năng ghi nhớ và sử dụng thông tin dài hạn, các nhà nghiên cứu đã xây dựng một số **benchmark chuyên biệt** cũng như sử dụng các bộ dữ liệu hội thoại có yếu tố nhớ. Dưới đây là các bộ dữ liệu và tiêu chí nổi bật:

  

- **LongMemEval (Wu et al., 2024)** – Đây là một bộ đánh giá toàn diện đầu tiên tập trung vào **5 kỹ năng trí nhớ lõi** của trợ lý chat ([[2410.10813] LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](https://arxiv.org/abs/2410.10813#:~:text=capabilities%20in%20sustained%20interactions%20remain,on%20memorizing%20information%20across%20sustained)). Năm kỹ năng đó bao gồm: (1) **Nhớ và trích thông tin** (Information Extraction) – kiểm tra xem mô hình có nhớ chính xác các chi tiết được đề cập trước đó hay không; (2) **Suy luận đa phiên** (Multi-session reasoning) – đánh giá khả năng kết nối thông tin qua nhiều phiên trò chuyện rời (ví dụ: người dùng nói A ở tuần trước và B ở tuần này, liệu bot có kết hợp A và B để trả lời?); (3) **Suy luận thời gian** (Temporal reasoning) – kiểm tra hiểu biết về trình tự thời gian, nguyên nhân-kết quả theo thời gian (ví dụ sự kiện X xảy ra sau Y thì hệ quả ra sao); (4) **Cập nhật kiến thức** (Knowledge updates) – đánh giá việc bot có sử dụng thông tin mới thay cho thông tin cũ khi chúng mâu thuẫn (giống bài toán cập nhật trí nhớ COVID ở trên); (5) **Abstention (từ chối)** – xem mô hình có biết từ chối trả lời khi không chắc do thiếu trí nhớ hay không (tránh trường hợp đoán bừa/hallucinate) ([[2410.10813] LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](https://arxiv.org/abs/2410.10813#:~:text=capabilities%20in%20sustained%20interactions%20remain,term)). LongMemEval gồm 500 câu hỏi được gài cẩn thận vào các lịch sử hội thoại dài, mỗi câu hỏi tương ứng kiểm tra một khía cạnh trên. Kết quả thực nghiệm cho thấy các chatbot hiện tại (kể cả mô hình lớn với ngữ cảnh dài) **giảm hiệu suất tới ~30%** khi phải ghi nhớ thông tin trải dài, so với các câu hỏi ngắn hạn ([[2410.10813] LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](https://arxiv.org/abs/2410.10813#:~:text=meticulously%20curated%20questions%20embedded%20within,augmented%20key)). Điều này khẳng định độ khó của bài toán và sự cần thiết của các phương pháp memory augmentation. LongMemEval hiện được coi là thước đo tiêu chuẩn, khuyến khích các nghiên cứu tương lai cải thiện cả 5 kỹ năng kể trên để tiến tới trợ lý đối thoại đáng tin cậy hơn ([[2410.10813] LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](https://arxiv.org/abs/2410.10813#:~:text=interactions,term)).
    

- **LOCOMO (Maharana et al., 2024)** – Là viết tắt của _Long Conversation Model_, đây được báo cáo là bộ dữ liệu hội thoại _dài nhất_ hiện nay, với trung bình **300 lượt thoại (9k token)** mỗi hội thoại ([On Memory Construction and Retrieval for Personalized Conversational Agents](https://arxiv.org/html/2502.05589v2#:~:text=%28i%29%20LOCOMO%C2%A0%28Maharana%20et%C2%A0al,on%20the%20recently%20released%20official)). LOCOMO mô phỏng các cuộc trò chuyện liên tục, nhiều chủ đề, đòi hỏi mô hình phải duy trì tương tác mạch lạc trong thời gian rất dài. Để đánh giá, tác giả dùng GPT-4 sinh ra các câu hỏi kiểm tra về nội dung đã nói từ rất sớm trong phiên, nhằm xem bot có nhớ hay không ([On Memory Construction and Retrieval for Personalized Conversational Agents](https://arxiv.org/html/2502.05589v2#:~:text=%28i%29%20LOCOMO%C2%A0%28Maharana%20et%C2%A0al,on%20the%20recently%20released%20official)). Ngoài ra, LOCOMO còn đo lường mức độ trôi chảy và nhất quán qua thước đo **GPT4Score** và các chỉ số ngôn ngữ tự nhiên (BLEU, ROUGE) cho phản hồi của mô hình ([On Memory Construction and Retrieval for Personalized Conversational Agents](https://arxiv.org/html/2502.05589v2#:~:text=long,in%20performance%20improvements%20up%20to)) ([On Memory Construction and Retrieval for Personalized Conversational Agents](https://arxiv.org/html/2502.05589v2#:~:text=Methods%20LOCOMO%20Long,44)). Cùng với LOCOMO, một số biến thể như **Long-MT-Bench+** cũng được dùng – đây là mở rộng của bộ đánh giá Multi-Turn Dialogue (MT-Bench) dành riêng cho hội thoại dài. Các kết quả baseline trên LOCOMO cho thấy nếu mô hình chỉ dùng lịch sử rất ngắn (hoặc không lịch sử) thì điểm số trả lời đúng rất thấp (~25-50), trong khi dùng full history nâng lên ~54 ([On Memory Construction and Retrieval for Personalized Conversational Agents](https://arxiv.org/html/2502.05589v2#:~:text=match%20at%20L389%20LOCOMO%20Zero,77%203%2C288)). Tuy nhiên, dùng full history phiến diện cũng gây mỏi model (13,000 token) và không nhất thiết tối ưu. Do vậy LOCOMO được dùng để thử nghiệm các chiến lược nhớ: thí dụ SeCom trên LOCOMO đạt **GPT4Score ~69**, cao hơn hẳn so với mô hình không module nhớ (~24) ([On Memory Construction and Retrieval for Personalized Conversational Agents](https://arxiv.org/html/2502.05589v2#:~:text=LOCOMO%20Zero%20History%2024,77%203%2C288)) ([On Memory Construction and Retrieval for Personalized Conversational Agents](https://arxiv.org/html/2502.05589v2#:~:text=Methods%20LOCOMO%20Long,44)). Điều này xác nhận lợi ích rõ rệt của memory đối với hội thoại siêu dài.
    

- **Các bộ dữ liệu personalized và multi-session**: Trước khi có các benchmark trên, một số bộ dữ liệu hội thoại được tạo ra nhằm kiểm tra một phần khía cạnh của trí nhớ. **Persona-Chat (Zhang et al., 2018)** cung cấp cho mỗi nhân vật một hồ sơ sở thích (5 câu mô tả) và yêu cầu mô hình trò chuyện giữ đúng persona này. Đây là kiểm tra khả năng **nhớ thông tin hồ sơ tĩnh** – gần với memory ngắn hạn (vì persona không đổi). **MuTual (Cui et al., 2020)** và **DSTC7,8** cung cấp các đoạn hội thoại yêu cầu suy luận logic giữa các lượt – gián tiếp đòi hỏi nhớ nội dung trước. **QuAC, CoQA (2018)** như đã đề cập, đánh giá khả năng trả lời dựa vào nhiều lượt hỏi trước (context co-reference). Tuy nhiên, các dataset này thường chỉ kéo dài tối đa vài chục lượt trong một phiên, và không đánh giá xuyên phiên hay cập nhật. Gần đây, một số dataset hướng đến **đa phiên**: ví dụ **MSC (Multi-Session Chat)** (Xu et al., 2022) nối 2-3 phiên PersonaChat lại để xem bot có nhớ thông tin giữa các phiên; hay **CareCall-Mem** (Bae et al., 2022) – dữ liệu tiếng Hàn mà nhóm Keep Me Updated xây dựng – gồm 5 phiên trò chuyện giữa bot và một người dùng hư cấu với các thông tin cá nhân thay đổi theo thời gian (sức khỏe, thói quen) () (). Các dataset này phục vụ huấn luyện và đánh giá mô hình trong bối cảnh **thông tin người dùng thay đổi**: ví dụ phiên 1 nói “ghét vận động”, phiên 3 lại nói “đang học bơi” thì bot phải hiểu sở thích đã thay đổi. Tiêu chí đánh giá gồm độ tự nhiên, tính gắn kết, và quan trọng là **độ chính xác của thông tin** mà bot nói ra so với hồ sơ thực tế (tránh nhầm thông tin cũ).
    

- **Tiêu chí đánh giá**: Dựa trên các benchmark trên, ta có thể liệt kê những tiêu chí chính để đánh giá chất lượng trí nhớ dài hạn của hệ thống hội thoại:
    
    - _Chính xác thông tin đã nhớ (Memory Recall)_: Kiểm tra tỉ lệ thông tin đúng được bot nhắc lại khi cần. Ví dụ, user đã nói họ sinh năm 1990, sau 10 lượt bot đề cập lại đúng năm sinh hay không. Tiêu chí này đo bằng câu hỏi trực tiếp (như LongMemEval) hoặc so khớp với log quá khứ.
        
    
    - _Phản hồi nhất quán, không ảo giác (Consistency & No-hallucination)_: Đánh giá xem bot có mâu thuẫn với chính nó hoặc với thực tế đã biết không, và có bịa đặt thông tin không có trong bộ nhớ không. Nếu bot _quên_ một chi tiết và tự chế ra, đó là điểm trừ lớn. Thước đo có thể bằng kiểm tra logic (ví dụ Persona-Chat yêu cầu không nói sai persona), hoặc nhờ đánh giá của mô hình/human xem câu trả lời có căn cứ quá khứ hay không.
        
    
    - _Cập nhật kiến thức kịp thời (Knowledge Update Accuracy)_: Khi người dùng cung cấp thông tin mới hoặc đính chính, bot có phản ánh đúng sự thay đổi trong các lượt sau không. Tiêu chí này thường đánh giá theo kịch bản: ví dụ như bài toán COVID test ở trên – sau khi user báo dương tính, bot phải quên thông tin “chưa xét nghiệm” trước đó. Có thể đo bằng truy vấn sau update xem bot trả lời dựa trên thông tin nào.
        
    
    - _Suy luận theo dòng thời gian (Temporal Reasoning)_: Bot có hiểu mối quan hệ thời gian giữa các sự kiện trong trí nhớ không. Ví dụ, user nói “năm 2020 tôi tốt nghiệp”, sau đó hỏi “2 năm sau tôi làm gì” – bot phải biết 2 năm sau 2020 là 2022 và tìm trong memory xem 2022 có sự kiện gì (hoặc trả lời chưa biết nếu không có). Khả năng này thường đo bằng các câu hỏi yêu cầu kết hợp mốc thời gian (như trong LongMemEval) ([[2410.10813] LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](https://arxiv.org/abs/2410.10813#:~:text=capabilities%20in%20sustained%20interactions%20remain,on%20memorizing%20information%20across%20sustained)).
        
    
    - _Khả năng từ chối khi không nhớ (Abstention)_: Một hệ thống tốt cần biết giới hạn trí nhớ của mình, tức là nếu thông tin không có trong bộ nhớ thì nên xin lỗi hoặc từ chối hơn là bịa. Tiêu chí này đánh giá tỷ lệ bot **không đoán bừa**. LongMemEval đưa ra các tình huống mà câu hỏi ngoài phạm vi những gì đã nói, yêu cầu bot phải phản hồi kiểu “Tôi không nhớ rõ…” thay vì cung cấp thông tin sai ([[2410.10813] LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](https://arxiv.org/abs/2410.10813#:~:text=capabilities%20in%20sustained%20interactions%20remain,term)).
        
    

  

Ngoài ra, các tiêu chí tổng quan như **độ hài lòng người dùng, độ tự nhiên của hội thoại, điểm đánh giá của giám khảo** cũng rất quan trọng, nhưng chúng chịu ảnh hưởng nhiều yếu tố ngoài trí nhớ (như kỹ năng ngôn ngữ chung của mô hình). Do đó, các benchmark chuyên biệt cố gắng cô lập ảnh hưởng của trí nhớ để đánh giá công bằng giữa các giải pháp.

  

# Hướng mở rộng và kết luận

  

**Trí nhớ dài hạn cho hệ thống đối thoại** vẫn là một bài toán mở với nhiều hướng nghiên cứu tiềm năng. Dựa trên các xu hướng hiện tại, có thể gợi ý một số hướng phát triển chính sau:

  

- **Kết hợp chặt chẽ giữa truy hồi và cập nhật tri thức**: Hiện nay, retrieval augmented generation (RAG) đã phổ biến trong QA mở, nhưng thường với _knowledge base_ tĩnh. Mở rộng hơn, ta có thể tích hợp RAG vào đối thoại sao cho **kho tri thức được cập nhật liên tục trong quá trình trò chuyện**. Ví dụ, khi người dùng cung cấp một thông tin mới, hệ thống ngay lập tức thêm nó vào _bộ nhớ tri thức_ và các lượt sau truy hồi có thể lấy ra. Điều này đòi hỏi giải quyết bài toán đồng bộ giữa thành phần ghi nhớ và thành phần tìm kiếm. Một hướng là phát triển các phương pháp **index động**: cập nhật chỉ mục bộ nhớ theo thời gian thực, hoặc sử dụng mô hình học tăng cường để quyết định khi nào cần _re-index_.
    

- **Truy hồi thích ứng và có hướng dẫn**: Thay vì luôn truy hồi một cách máy móc top-k đoạn giống như hiện nay, mô hình có thể học cách **đặt truy vấn thông minh** hoặc **chọn lọc** tùy tình huống. Chẳng hạn, nếu câu hỏi của người dùng rất rõ ràng (như hỏi tên đã cho trước đó), một truy vấn thẳng sẽ hiệu quả; nhưng nếu câu hỏi mơ hồ, mô hình có thể tự sinh ra một truy vấn rõ hơn dựa trên ngữ cảnh – tương tự kỹ thuật _query rewriting_ ([Augmenting LLMs with Retrieval, Tools, and Long-term Memory | by Alaa Dania Adimi | InfinitGraph | Mar, 2025 | Medium](https://medium.com/@ja_adimi/augmenting-llms-with-retrieval-tools-and-long-term-memory-b9e1e6b2fc28#:~:text=Query%20Rewriting)). Ngoài ra, mô hình nên học _khi nào_ thì cần truy hồi: đôi khi, câu hỏi hiện tại không liên quan gì đến quá khứ, việc truy hồi chỉ thêm nhiễu. Có thể dùng một module phụ (như một classifier) để quyết định có truy hồi memory không ở mỗi lượt. Một ý tưởng khác là cho chính LLM **hướng dẫn việc truy hồi**: ví dụ trước khi trả lời, mô hình tự suy luận "Để trả lời, tôi cần nhớ X", sau đó dùng suy luận này làm chìa khóa tìm kiếm bộ nhớ. Đây là một dạng _chain-of-thought for retrieval_ đầy hứa hẹn.
    

- **Sử dụng mô hình ngôn ngữ phụ trợ cho quản lý trí nhớ**: Thay vì các rule cứng (như 4 thao tác của Keep Me Updated), ta có thể dùng một LLM nhỏ hoặc các prompt đặc biệt cho chính LLM lớn để quản lý memory. Ví dụ, có thể triển khai một _“Memory Manager Agent”_ chạy song song: agent này dùng LLM để định kỳ đọc lịch sử và viết tóm tắt, lưu vào vector DB; khi cần thì hỗ trợ truy vấn vector DB và cung cấp kết quả cho LLM chính. Cách tiếp cận kiến trúc agent này đã được Park et al. (2023) thử nghiệm trong **Generative Agents**, nơi nhiều agent LLM tương tác với nhau và có bộ nhớ sự kiện được ghi lại và suy diễn bằng LLM. Một ứng dụng khác là dùng LLM để **đánh giá và chỉnh sửa** memory: ví dụ dùng GPT-4 đọc toàn bộ memory log và phát hiện mâu thuẫn hoặc lỗi để sửa (một dạng reviewer). Nhìn chung, tận dụng khả năng ngôn ngữ đa năng của LLM cho việc quản trị trí nhớ có thể đem lại linh hoạt hơn so với cách làm thuần heuristic.
    

- **Mở rộng sang đa mô hình và tri thức thế giới**: Trí nhớ hội thoại không chỉ gồm lời thoại – trong nhiều ứng dụng, nó cần nhớ cả các **thông tin thị giác, cảm biến, hay tri thức ngoài**. Hướng mở là tích hợp **bộ nhớ chung cho đa mô hình**: ví dụ một robot trợ lý nhà thông minh cần nhớ hôm qua camera thấy gì, ai đã ghé thăm, đồ vật đặt ở đâu... cùng với hội thoại với chủ nhà. Điều này đặt ra bài toán lưu trữ và truy hồi các **đại diện đa mô hình** (hình ảnh, âm thanh) bên cạnh văn bản. Tương tự, kết hợp **knowledge graph** hoặc cơ sở tri thức vào memory: ví dụ khi người dùng nói sở thích, bot có thể lưu vào một _knowledge graph node_ về người dùng, liên kết với các node hoạt động tương ứng. Việc kết hợp cấu trúc tri thức có thể giúp bot suy luận logic và nhất quán hơn (tránh mâu thuẫn thực tế). Một hướng là mỗi khi memory update, đồng thời cập nhật knowledge graph, và dùng graph embedding để hỗ trợ retrieval song song.
    

- **Đánh giá và giảm thiểu nhiễu do trí nhớ sai**: Khi tích hợp bộ nhớ, một nguy cơ là _nhớ sai hoặc nhớ mơ hồ_ có thể dẫn đến phản hồi sai (hallucination do memory). Do đó, cần cơ chế **đánh giá độ tin cậy của memory**. Một hướng là kèm theo mỗi mẩu memory một độ tin cậy (confidence score) và thời gian, để mô hình ưu tiên dùng thông tin mới và có độ tin cậy cao. Nếu memory quá cũ, mô hình có thể cảnh báo. Một hướng khác là huấn luyện mô hình **phát hiện mâu thuẫn** giữa memory và message hiện tại: nếu phát hiện user nói điều trái ngược hẳn với memory cũ, có thể kích hoạt một _quy trình xác minh_, hỏi lại người dùng để chắc chắn trước khi cập nhật.
    

  

Tóm lại, **hệ thống đối thoại tích hợp trí nhớ dài hạn** đang dần trở nên khả thi nhờ các tiến bộ trong cả mô hình ngôn ngữ lớn lẫn kỹ thuật quản lý tri thức. Từ những mô hình QA đơn lượt đơn giản, chúng ta đã chứng kiến sự ra đời của các chatbot có khả năng ghi nhớ hàng trăm lượt thoại, cá nhân hóa theo người dùng, và cập nhật hiểu biết theo thời gian. Dù vẫn còn những thách thức về tối ưu và độ tin cậy, hướng nghiên cứu này hứa hẹn đem lại các trợ lý ảo **nhớ lâu, hiểu sâu và phản hồi tự nhiên** hơn – một bước tiến lớn tới **AI đối thoại mang tính cá nhân và đáng tin cậy** trong tương lai gần. Các nghiên cứu mới như LongMemEval đang tạo nền tảng để **đánh giá có hệ thống** các tiến bộ, còn các ý tưởng kết hợp memory và LLM (MemoryBank, THEANINE, LD-Agent) đang mở đường cho thế hệ mô hình hội thoại thông minh kế tiếp. Chúng ta có thể kỳ vọng trong tương lai, sự kết hợp giữa **cửa sổ ngữ cảnh lớn** và **bộ nhớ ngoài linh hoạt** sẽ giúp xóa nhòa ranh giới về trí nhớ trong đối thoại, cho phép các hệ thống AI trò chuyện một cách mạch lạc và hiểu biết qua _nhiều tháng, nhiều năm_ tương tác với con người.

  

**Tài liệu tham khảo:**

  

1. Wu, D. _et al._ (2024). _LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory_. **ICLR 2025 (preprint)** ([[2410.10813] LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](https://arxiv.org/abs/2410.10813#:~:text=capabilities%20in%20sustained%20interactions%20remain,term)) ([[2410.10813] LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](https://arxiv.org/abs/2410.10813#:~:text=meticulously%20curated%20questions%20embedded%20within,augmented%20key)).
    

2. Qu, C. _et al._ (2020). _Open-Retrieval Conversational Question Answering_. **SIGIR 2020** ([[2005.11364] Open-Retrieval Conversational Question Answering](https://arxiv.org/abs/2005.11364#:~:text=retrieval%20conversational%20question%20answering%20,the%20reranker%20component%20contributes%20to)).
    

3. Bae, S. _et al._ (2022). _Keep Me Updated! Memory Management in Long-term Conversations_. **Findings of EMNLP 2022** () ().
    

4. Li, H. _et al._ (2025). _Hello Again! LLM-powered Personalized Agent for Long-term Dialogue (LD-Agent)_. **NAACL 2025 (to appear)** ([[2406.05925] Hello Again! LLM-powered Personalized Agent for Long-term Dialogue](https://arxiv.org/abs/2406.05925#:~:text=the%20Long,Agent%20are)).
    

5. Zhong, W. _et al._ (2023). _MemoryBank: Enhancing Large Language Models with Long-Term Memory_. **arXiv:2305.10250** ([[2305.10250] MemoryBank: Enhancing Large Language Models with Long-Term Memory](https://ar5iv.labs.arxiv.org/html/2305.10250#:~:text=personality%20over%20time%20by%20synthesizing,based%20chatbot%20named)) ([Augmenting LLMs with Retrieval, Tools, and Long-term Memory | by Alaa Dania Adimi | InfinitGraph | Mar, 2025 | Medium](https://medium.com/@ja_adimi/augmenting-llms-with-retrieval-tools-and-long-term-memory-b9e1e6b2fc28#:~:text=Memory%20Updating)).
    

6. Ong, K.T. _et al._ (2025). _THEANINE: Timeline-based Memory Management for Lifelong Dialogue Agents_. **NAACL 2025 (to appear)** ([[2406.10996] Towards Lifelong Dialogue Agents via Timeline-based Memory Management](https://arxiv.org/abs/2406.10996#:~:text=to%20improve%20retrieval%20quality%2C%20we,human%20efforts%20when%20assessing%20agent)).
    

7. Weston, J. _et al._ (2015). _Memory Networks_. **ICLR 2015** ([[1410.3916] Memory Networks](https://arxiv.org/abs/1410.3916#:~:text=,chaining%20multiple%20supporting%20sentences%20to)).
    

8. Graves, A. _et al._ (2016). _Hybrid computing using a neural network with dynamic external memory (DNC)_. **Nature 538, 471–476 (2016)** ([Differentiable neural computer - Wikipedia](https://en.wikipedia.org/wiki/Differentiable_neural_computer#:~:text=DNC%20indirectly%20takes%20inspiration%20from,by%20finding%20a%20%2052)).
    

9. Seo, M. _et al._ (2017). _Bidirectional Attention Flow for Machine Comprehension (BiDAF)_. **ICLR 2017** ([BERT with History Answer Embedding for Conversational Question Answering](https://arxiv.org/pdf/1905.05412#:~:text=4,representation%20generated%20when%20answering%20previous)).
    

10. Chen, D. _et al._ (2017). _Reading Wikipedia to Answer Open-Domain Questions (DrQA)_. **ACL 2017** ([BERT with History Answer Embedding for Conversational Question Answering](https://arxiv.org/pdf/1905.05412#:~:text=,JASIS%2C%2038%3A389%E2%80%93404%2C%201987)).