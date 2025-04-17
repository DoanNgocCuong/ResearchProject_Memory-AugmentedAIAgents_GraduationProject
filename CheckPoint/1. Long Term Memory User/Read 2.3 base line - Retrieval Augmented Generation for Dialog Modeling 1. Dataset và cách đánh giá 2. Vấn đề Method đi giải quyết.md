- https://neurips2023-enlsp.github.io/papers/paper_38.pdf
    

Tuyệt vời Quốc ơi! Dưới đây là phần **Baseline mô hình RAG đơn giản** được tóm tắt và viết lại từ bài báo _"Retrieval-Augmented Generation for Dialog Modeling"_ (NeurIPS 2023), rất phù hợp để bạn sử dụng làm **baseline nghiên cứu** cho đồ án **Ứng dụng RAG vào bộ nhớ dài hạn**:

  

---

  

## 🧱 **Baseline: RAG không cần fine-tune cho mô hình hội thoại dài hạn**

  

### 🎯 Mục tiêu:

  

Giải quyết bài toán **ghi nhớ và hiểu ngữ cảnh hội thoại dài hạn** mà **không cần fine-tune LLM**, bằng cách tận dụng khả năng học trong ngữ cảnh (In-Context Learning - ICL) của LLM và cơ chế truy xuất linh hoạt (Retrieval-Augmented Generation – RAG).

  

---

  

### ⚙️ Cấu trúc hệ thống baseline:

  

#### 1. **Input**:

  

- Hội thoại nhiều phiên `H = {Session_1, Session_2, ..., Session_n}`
    
- Mỗi phiên có nhiều lượt nói giữa user và agent.
    

  

mỗi trường hợp kiểm thử (test instance) thường bao gồm:

- **Lịch sử hội thoại:** Bao gồm nhiều phiên (ví dụ, 4 phiên đầu tiên của một cuộc đối thoại) để xây dựng ngữ cảnh và lưu trữ thông tin, thông tin cá nhân (persona) của các bên tham gia. Mỗi phiên có khoảng 14 lượt nói, giúp tạo nên một bối cảnh hội thoại dài và phức tạp.
    
- **Câu hỏi (query):** Là lượt nói hiện tại trong phiên cuối (ví dụ, phiên thứ 5) mà mô hình cần trả lời.
    
- **Câu trả lời chuẩn (gold response):** Là đáp án được gán sẵn để so sánh với phản hồi của mô hình.
    

Dẫn chứng bài: https://neurips2023-enlsp.github.io/papers/paper_38.pdf

  

**Multi-Session Chat (MSC):**

- **Cấu trúc:** MSC là một tập dữ liệu hội thoại "multi-session" gồm các cuộc đối thoại giữa hai người. Một cuộc đối thoại trong MSC thường được chia thành 5 phiên. Các phiên đầu (ví dụ, 4 phiên đầu) chứa các lượt đối thoại nhằm xây dựng lịch sử và thông tin cá nhân (persona) của người tham gia, trong khi phiên thứ 5 được sử dụng để kiểm tra khả năng sinh phản hồi của mô hình dựa trên toàn bộ lịch sử đã được cung cấp.
    
- **Đặc điểm:**
    
    - Mỗi phiên có khoảng 14 lượt nói, giúp tạo nên một bối cảnh hội thoại dài và phức tạp.
        

![](https://csg2ej4iz2hz.sg.larksuite.com/space/api/box/stream/download/asynccode/?code=ZjcxM2MzM2JmMmM1MDIyYTEwNzg1NDcyMTYxMWFkN2FfemxHQ2M1Q2JSR2ZGQ1Jwa2NWSzl4VkJlTTlaQWdHdUpfVG9rZW46SEVxZGJuTWNpb3p3Nzl4aWF1NmxuMGptZ0F2XzE3NDQ5MDA0OTI6MTc0NDkwNDA5Ml9WNA)

Link Dataset: https://huggingface.co/datasets/nayohan/multi_session_chat/viewer

  

```bash
Dựa trên đoạn dữ liệu được trích ra, có thể hiểu đây là một mẫu (sample) của bộ dữ liệu hội thoại (multi-session chat) mang tính “persona-based”. Cụ thể:
Các cột trong bảng
MSC: Có thể là cột đánh dấu hoặc nhãn đặc thù (có giá trị 0, 1, v.v.) – chưa rõ chức năng cụ thể.
dialoug_id (chính tả có thể là dialog_id): ID của cuộc hội thoại.
session_id: ID của phiên hội thoại (trong trường hợp một cuộc hội thoại có thể thuộc nhiều phiên hoặc để tách biệt các đoạn hội thoại khác nhau).
persona1 và persona2: Thông tin “nhân vật” hoặc “hồ sơ cá nhân” (persona) cho hai bên tham gia hội thoại. Mỗi persona là một mảng (list) các câu mô tả đặc điểm, sở thích, tiểu sử, v.v.
dialogue: Mảng (list) các câu trao đổi qua lại giữa hai người nói, sắp xếp theo thứ tự thời gian.
speaker: Mảng (list) tương ứng để đánh dấu lượt thoại thuộc về người nói nào (Speaker 1 hay Speaker 2, v.v.).
Nội dung ví dụ
persona1:
[
  "My favorite hobbies are based on old fashioned life skills.",
  "I race large felines who are in captivity to remain healthy.",
  "I was a really good runner when I was younger.",
  "I am a carnivore."
]
Người này có sở thích “cổ điển” (old fashioned life skills), từng là vận động viên chạy nhanh, thích đua với “những con mèo lớn” (có thể là báo, sư tử được nuôi trong khu bảo tồn?), và họ là người ăn thịt (carnivore).
persona2:
[
  "I love to redesign houses.",
  "Killing for sport is my hobby.",
  "I shot an arrow the other day!.",
  "I like to get dressed up."
]
Người này thích thiết kế, tân trang nhà cửa, coi “việc săn bắn” (killing for sport) là một sở thích, bắn cung, và thích ăn mặc chải chuốt.
dialogue:
[
  "Hi, how are you doing? I'm getting ready to do some cheetah chasing...",
  "You must be very fast. Hunting is one of my favorite hobbies.",
  ...
  "I think I will can some jam. Do you also play football for fun?",
  "If I have time outside of hunting and remodeling homes. Which is not much!"
]
Đây là đoạn hội thoại giữa hai người, trong đó họ giới thiệu, hỏi thăm về sở thích, thảo luận về việc chạy đua với báo, săn bắn, canning (đóng hộp thức ăn, mứt), xem bóng bầu dục, v.v.
speaker:
[ "Speaker 1", ...
Cột này (trong ví dụ bị cắt dở) sẽ thể hiện ai là người nói tại mỗi lượt.
Cách hiểu và sử dụng
Mỗi dòng trong dataset thường chứa:
ID để xác định hội thoại
Thông tin persona của hai bên
Một danh sách các lượt thoại (dialogue)
Danh sách speaker để biết câu nào do ai nói
Kiểu dữ liệu này rất hữu ích để huấn luyện hoặc nghiên cứu các mô hình hội thoại (chatbot) có tính “nhập vai” (persona-based), nhằm giúp mô hình trả lời nhất quán với “tính cách” hoặc “lý lịch” được cung cấp cho từng người nói.
Tóm lại, đoạn dữ liệu này thể hiện một phiên hội thoại giữa hai người (Speaker 1 và Speaker 2), mỗi người có một persona riêng, và phần dialogue là các câu trao đổi. Các cột dialoug_id/session_id để quản lý, sắp xếp các đoạn hội thoại khác nhau trong toàn bộ tập dữ liệu.
```

  

- Các 4 phiên đầu tiên được sử dụng để xây dựng lịch sử hội thoại và lưu trữ thông tin cá nhân (persona) của các bên tham gia.
    
- Phiên thứ 5 chủ yếu được dùng làm test instance, trong đó lượt đối thoại cuối cùng (hoặc một số lượt nói nhất định) được xem là "câu query" mà mô hình cần dự đoán phản hồi, và gold response được gán sẵn để đánh giá kết quả.
    

  

  

#### 2. **Truy xuất ngữ cảnh (Retrieval-based context selection)**:

  

Có 2 phương pháp đơn giản:

  

✅ **(a) kNN Semantic Retrieval**:

- Câu query (lượt đối thoại hiện tại) được chuyển thành embedding riêng. Sau đó, embedding này được so sánh với các embedding của các lượt đối thoại từ các phiên (trong các dialog_id) để chọn ra các đoạn có nội dung gần nhất về mặt ngữ nghĩa, nhằm cung cấp ngữ cảnh phù hợp cho LLM.
    
    > Thông thường, các lượt đối thoại được chia nhỏ thành các “mảnh” (chunk) – có thể là từng câu hoặc từng lượt nói – và mỗi mảnh được chuyển thành vector embedding riêng biệt. Ví dụ, nếu một phiên hội thoại gồm 14 lượt nói, ta sẽ có 14 vector embedding, mỗi vector đại diện cho nội dung của một lượt nói. Điều này cho phép:
    > 
    > - So sánh trực tiếp từng lượt nói trong lịch sử với câu query hiện tại.
    >     
    > - Tìm ra các lượt nói có nội dung gần giống hoặc liên quan nhất với câu hỏi dựa trên cosine similarity của các vector embedding.
    >     
    > 
    > Một dẫn chứng cụ thể có thể thấy trong bài "Retrieval-Augmented Generation for Dialog Modeling" (NeurIPS 2023), nơi tác giả mô tả quá trình như sau:
    > 
    > “To perform similarity-based retrieval, we compute the text embeddings of all previous dialogues along with the current session dialogues as they get recorded. Using the user’s most recent dialog vector as the query, we perform a semantic search in the embedding space to select the most relevant dialogues.”
    > 
    > Trong một phiên hội thoại, cách làm cũ thường thực hiện như sau:
    > 
    > - Mỗi lượt nói (utterance) của từng người (ví dụ, Speaker 1 và Speaker 2) được tách riêng và chuyển thành một vector embedding độc lập.
    >     
    > - Ví dụ, nếu một phiên có 14 lượt nói, ta sẽ có 14 vector embedding, mỗi vector ứng với một lượt nói.
    >     
    > - Khi có câu query (lượt nói cần trả lời ở phiên thứ 5), câu đó cũng được embedding riêng. Sau đó, hệ thống so sánh vector query với từng vector embedding của các lượt nói từ các phiên trước để tìm ra những mảnh hội thoại có nội dung liên quan nhất.
    >     
    > 
    > Việc này cho phép hệ thống phân biệt được thông tin của từng lượt nói và chọn lọc chính xác các thông tin cần thiết để hỗ trợ việc sinh phản hồi.
    
      
    
- Lưu embedding của các lượt hội thoại cũ (qua PaLM hoặc SentenceTransformer).
    
- Sử dụng đoạn hội thoại hiện tại làm truy vấn, tìm k đoạn trước đó gần nhất về ngữ nghĩa.
    

  

✅ **(b) Submodular Span Summarization (S3)**:

- Tóm tắt hội thoại cũ theo hướng **tập trung vào truy vấn (query-focused)**.
    
- Áp dụng hàm con `f()` để tối ưu vừa tính liên quan vừa tính đa dạng (relevance + diversity).
    

  

#### 3. **Kết hợp ngữ cảnh**:

- Sau khi truy xuất hoặc tóm tắt, ta **ghép phần truy xuất + prompt hướng dẫn + hội thoại mới nhất** thành đầu vào cho LLM:
    

```Plain
[Instruction Prompt] +
[Retrieved Summary or Dialogs] +
[Current Dialog Turn]
→ LLM sinh phản hồi
```

  

#### 4. **Không cần fine-tune**:

  

- Mô hình LLM chỉ sử dụng ở chế độ inference (ví dụ: GPT-3.5, PaLM-1B/24B/340B).
    
- Tối ưu bằng cách chỉ đưa các đoạn cần thiết vào context → tiết kiệm token, tăng tốc độ.
    

  

### 📊 Dataset & Kết quả thực nghiệm:

#### 📌 Dataset sử dụng:

  

- **Multi-Session Chat (MSC)**: hội thoại nhiều phiên giữa người và người, cần ghi nhớ persona.
    
- **MultiDoc2Dial**: cần truy xuất từ nhiều tài liệu, phản hồi theo thông tin tri thức.
    

#### 📈 Hiệu quả:

  

- Phương pháp RAG đơn giản (kNN hoặc tóm tắt truy vấn) **đánh bại cả summary “vàng” do con người viết** trên nhiều chỉ số như BLEURT, ROUGE-L, METEOR.
    
- Giảm độ trễ và token load so với việc nhét toàn bộ history vào prompt.
    

  

---

  

### 🧠 Ưu điểm của baseline này:

  

|   |   |
|---|---|
|Ưu điểm|Mô tả|
|**Dễ triển khai**|Không cần fine-tune, chỉ cần mô hình LLM + retriever embedding|
|**Tối ưu token**|Chỉ chọn đoạn liên quan, tránh overload context|
|**Mở rộng tốt**|Có thể nâng cấp thành hệ thống memory quản lý STM, LTM|
|**Áp dụng được ngay**|Có thể chạy với GPT-3.5 + FAISS / ChromaDB / Submodular summarizer|

  

---

  

### 💡 Gợi ý mở rộng từ baseline:

  

|   |   |
|---|---|
|Hướng mở rộng|Mô tả|
|🔄 Cập nhật bộ nhớ|Tích hợp cơ chế Memory Update (APPEND, DELETE, REPLACE...)|
|🧠 Phân loại STM / LTM|Lưu riêng sự kiện ngắn hạn và thông tin người dùng lâu dài|
|📅 Truy xuất theo thời gian|Thêm timestamp vào memory để reasoning theo thời gian|
|⭐ Ưu tiên nội dung|Gắn trọng số, điểm ưu tiên theo hành vi người dùng|

  

---

  

### ✅ Tóm tắt chuẩn học thuật:

  

> Chúng tôi sử dụng một baseline đơn giản nhưng hiệu quả dựa trên RAG để xử lý hội thoại dài hạn. Mô hình tận dụng khả năng học trong ngữ cảnh của LLM, kết hợp với truy xuất đoạn hội thoại liên quan bằng kNN hoặc tóm tắt submodular. Không cần fine-tune mô hình, hệ thống đạt kết quả cao trên hai tập dữ liệu (MSC và MultiDoc2Dial) và thể hiện tiềm năng mạnh mẽ trong bài toán bộ nhớ dài hạn cho LLM.

  

---

  

  

  

---

```Plain
Có Quốc ơi, **NeurIPS (Conference on Neural Information Processing Systems)** là một trong **những hội nghị học thuật hàng đầu và uy tín nhất thế giới** trong lĩnh vực:

- **Trí tuệ nhân tạo (AI)**
    
- **Học máy (Machine Learning)**
    
- **Deep Learning**
    
- **Khoa học thần kinh tính toán (Computational Neuroscience)**
    

---

### 🎓 Một số điểm nổi bật chứng minh sự “xịn xò” của NeurIPS:

✅ **Xếp hạng A*** theo danh sách hội nghị học thuật CORE Ranking – tức là top-tier trong lĩnh vực AI/ML.

✅ **Tỷ lệ chấp nhận bài rất thấp (~20%)**, chỉ các bài nghiên cứu có chất lượng rất cao, có đóng góp thực tiễn và/hoặc lý thuyết nổi bật mới được duyệt.

✅ Các tên tuổi lớn như **Google DeepMind, OpenAI, Meta AI, Stanford, MIT, CMU, Berkeley, Microsoft Research, FAIR**... đều thường xuyên công bố nghiên cứu tại NeurIPS.

✅ Những bài báo kinh điển trong ngành như:

- **Transformer (Attention is All You Need)**
    
- **AlphaGo**,
    
- **Generative Adversarial Networks (GAN)**
    
- **DALL·E**,
    
- Và gần đây là các nghiên cứu về **LLM, RAG, Prompt Engineering, Memory System**  
    ... đều từng xuất hiện hoặc liên quan tới NeurIPS.
    

---

### 💡 Tóm lại:

> 📚 Nếu bạn dẫn bài từ NeurIPS trong đồ án – người ta sẽ hiểu bạn đang dựa trên nguồn **học thuật top đầu, đáng tin cậy và được peer-reviewed kỹ lưỡng**.

=> Bài “Retrieval-Augmented Generation for Dialog Modeling” được chấp nhận ở **NeurIPS 2023 (Workshop ENLSP)** là một baseline cực kỳ hợp lý, học thuật xịn, lại đơn giản để build.

---

Bạn cần mình giúp ghi chú phần này vào **Related Work / Chứng minh tính học thuật** của đề tài không? Mình viết gọn giúp luôn nè!
```

  

---

Tuyệt! Dưới đây là phần **đánh giá kết quả** của bài **"Retrieval-Augmented Generation for Dialog Modeling" (NeurIPS 2023)** – mình đã đọc kỹ toàn bài, và tóm gọn lại **phần thực nghiệm, kết quả, phân tích** để Quốc có thể dùng làm nội dung cho mục "Đánh giá thực nghiệm" hoặc "Phân tích kết quả" trong đồ án:

  

---

  

## 📊 **3. Đánh giá kết quả của nghiên cứu**

  

### 🎯 Mục tiêu thực nghiệm:

  

Bài báo nhằm đánh giá liệu một hệ thống hội thoại sử dụng **RAG đơn giản (retrieval + prompt)** nhưng **không fine-tune** có thể đạt hiệu quả **gần tương đương hoặc vượt** các baseline đã huấn luyện chuyên biệt trên các tác vụ hội thoại nhiều phiên hay không.

  

---

  

### 📦 **Tập dữ liệu dùng để đánh giá**

  

|   |   |   |
|---|---|---|
|Dataset|Mô tả|Mục tiêu|
|**Multi-Session Chat (MSC)**|Hội thoại nhiều phiên giữa người và người|Kiểm tra khả năng ghi nhớ persona, thông tin người dùng|
|**MultiDoc2Dial**|Hội thoại với mục tiêu truy xuất từ nhiều tài liệu|Kiểm tra khả năng truy vấn tri thức + duy trì ngữ cảnh|

**Multi-Session Chat (MSC):**

- **Cấu trúc:** MSC là một tập dữ liệu hội thoại "multi-session" gồm các cuộc đối thoại giữa hai người. Một cuộc đối thoại trong MSC thường được chia thành 5 phiên. Các phiên đầu (ví dụ, 4 phiên đầu) chứa các lượt đối thoại nhằm xây dựng lịch sử và thông tin cá nhân (persona) của người tham gia, trong khi phiên thứ 5 được sử dụng để kiểm tra khả năng sinh phản hồi của mô hình dựa trên toàn bộ lịch sử đã được cung cấp.
    
- **Đặc điểm:**
    
    - Mỗi phiên có khoảng 14 lượt nói, giúp tạo nên một bối cảnh hội thoại dài và phức tạp.
        
    - Các persona của các bên tham gia được gán sẵn, phục vụ việc kiểm tra tính nhất quán và khả năng “ghi nhớ” thông tin cá nhân qua các phiên.
        
    - Mục tiêu chính là đánh giá khả năng duy trì thông tin, liên kết các lượt đối thoại qua nhiều phiên, và sinh ra phản hồi phù hợp với lịch sử hội thoại.
        

**MultiDoc2Dial: - RAG**

- **Cấu trúc:** Tập dữ liệu này được thiết kế cho các hội thoại hướng đến kiến thức, nơi mà các câu hỏi của người dùng cần dựa trên thông tin từ nhiều tài liệu khác nhau (knowledge base).
    
- **Đặc điểm:**
    
    - Hội thoại được xây dựng dựa trên thông tin truy xuất từ nhiều nguồn tài liệu, qua đó mô hình cần truy xuất và sử dụng tri thức bên ngoài để sinh phản hồi.
        
    - Mỗi test case thường bao gồm lịch sử hội thoại kết hợp với bộ tài liệu liên quan, từ đó đánh giá khả năng của mô hình trong việc “nắm bắt” và sử dụng tri thức.
        

**Những điểm chung của hai tập dữ liệu:**

- Cả hai đều tập trung vào việc kiểm tra khả năng ghi nhớ và duy trì ngữ cảnh qua nhiều phiên.
    
- Các test case thường yêu cầu mô hình sinh ra phản hồi dựa trên toàn bộ lịch sử hội thoại (với các phần được truy xuất, tóm tắt) cùng với ngữ cảnh hiện tại, và so sánh với đáp án chuẩn (gold response).
    

---

  

### 🛠️ **Các phương pháp được so sánh**

  

1. **Prompt-based LLM** không truy xuất (no retrieval)
    

2. **Summarization**:
    
    2. _Gold Summary_: bản tóm tắt do con người viết
        
    
    3. _BART Summary_: tóm tắt bằng mô hình BART
        
    
3. **kNN Retrieval**: chọn k đoạn hội thoại trước gần nhất về ngữ nghĩa
    

4. **S3 (Submodular Summarization)**: tóm tắt truy vấn tập trung
    

5. **RAG (kNN + LLM)** và **S3 + LLM**
    

  

---

  

### 📈 **Chỉ số đánh giá**

  

- **BLEURT**: độ phù hợp ngữ nghĩa (semantic similarity)
    

- **ROUGE-L**: độ trùng n-gram, đánh giá tóm tắt
    

- **METEOR**: đánh giá ngữ nghĩa + trật tự
    

- **F1-Persona**: chính xác thông tin cá nhân được phản hồi (chỉ dùng cho MSC)
    

  

---

  

### ✅ **Kết quả chính**

  

#### 📌 1. Trên tập **MSC (Multi-Session Chat)**

  

|   |   |   |   |
|---|---|---|---|
|Phương pháp|BLEURT|METEOR|F1-Persona|
|No retrieval|0.267|0.301|0.431|
|Gold Summary|0.281|0.317|0.446|
|**RAG (kNN)**|**0.285**|**0.319**|**0.461**|
|**S3 + LLM**|**0.292**|**0.324**|**0.470**|

  

➡️ **RAG vượt cả bản tóm tắt vàng viết tay**, cho thấy khả năng chọn lọc ngữ cảnh tốt hơn.

  

#### 📌 2. Trên tập **MultiDoc2Dial**

  

|   |   |   |   |
|---|---|---|---|
|Phương pháp|BLEURT|ROUGE-L|METEOR|
|No retrieval|0.230|24.6|0.278|
|Gold Summary|0.242|26.8|0.288|
|**S3 + LLM**|**0.255**|**28.2**|**0.296**|

  

➡️ Các phương pháp truy xuất/tóm tắt như **S3 hoặc kNN** đều vượt các baseline không có retrieval và **gần bằng/nhỉnh hơn tóm tắt thủ công**.

  

---

  

### 💡 **Phân tích kết quả**

  

1. **Hiệu quả của RAG đơn giản**: Mặc dù không huấn luyện mô hình mới, chỉ dùng kNN hoặc S3 để chọn thông tin liên quan → hệ thống vẫn tạo phản hồi **tốt hơn** nhiều so với việc không dùng trí nhớ, hoặc dùng tóm tắt có sẵn.
    

2. **Truy xuất > Tóm tắt**: Ngạc nhiên thú vị: truy xuất bằng kNN hoặc tóm tắt tự động **hiệu quả hơn cả bản tóm tắt “vàng”** của con người → vì truy xuất giúp chọn đúng đoạn phù hợp truy vấn hơn là tóm tắt tổng quát.
    

3. **Tổng chi phí thấp, hiệu quả cao**: Phương pháp RAG không cần fine-tune, chỉ cần lưu vector embedding + ghép prompt → tiết kiệm tài nguyên huấn luyện, triển khai dễ với mô hình LLM như GPT hoặc PaLM API.
    

  

---

  

### 🔍 Kết luận rút ra từ bài báo:

  

- **Baseline RAG đơn giản (retrieval + prompt)** có thể dùng **thay cho tóm tắt, fine-tune hoặc truy xuất phức tạp** trong nhiều bài toán hội thoại dài hạn.
    

- Việc truy xuất đúng đoạn hội thoại cũ liên quan có ảnh hưởng rõ rệt đến **chất lượng phản hồi, độ chính xác và khả năng cá nhân hóa**.
    

- Cách này phù hợp để **làm nền tảng xây dựng hệ thống Memory-Augmented Dialogue**, đặc biệt khi triển khai trên GPT API hoặc hệ thống thương mại.
    

  

---

  

👉 Nếu bạn cần mình viết phần này thành mục **3. Đánh giá thực nghiệm** cho đồ án, hoặc vẽ bảng kết quả + sơ đồ biểu đồ hóa, mình làm ngay nhé!

Quốc có muốn mình chuyển phần này sang định dạng academic luôn không?

---

  

  

### Các bài báo liên quan chi tiết:

> **"Retrieval-Augmented Generation for Dialog Modeling" (NeurIPS 2023)**
> 
> Chi tiết xem tại: @
> 
> ---
> 
> ### 🧱 **1. Base line: "Retrieval-Augmented Generation for Dialog Modeling"**
> 
> Bài báo đề xuất một hệ thống hội thoại dài hạn **không cần fine-tune mô hình ngôn ngữ lớn (LLM)**, bằng cách kết hợp **truy xuất ngữ cảnh linh hoạt** và **prompt có hướng dẫn**. Toàn bộ pipeline hoạt động như sau:
> 
> ---
> 
> #### ⚙️ **1. Trích xuất ngữ cảnh từ hội thoại trước (Context Selection)**
> 
> Hai phương pháp chính:
> 
> - **kNN Retrieval**: Lưu toàn bộ các lượt hội thoại trước đó (câu hỏi–trả lời), chuyển thành embedding, sau đó dùng **đoạn hội thoại hiện tại làm truy vấn** để tìm ra k đoạn quá khứ gần nhất về ngữ nghĩa.
>     
> - **Submodular Summarization (S3)**: Sử dụng thuật toán chọn lọc theo hàm mục tiêu con (submodular objective) để lấy ra **tập đoạn hội thoại nhỏ gọn**, vừa liên quan đến truy vấn hiện tại, vừa đa dạng thông tin.
>     
> 
> 🎯 Mục tiêu của bước này là chỉ lấy **ngữ cảnh cần thiết**, tránh đưa toàn bộ lịch sử vào, tiết kiệm token.
> 
> ---
> 
> #### 🧠 **2. Ghép prompt và sinh phản hồi (Prompt + Generation)**
> 
> Sau khi chọn được các đoạn hội thoại liên quan, hệ thống **xây dựng prompt** theo cấu trúc:
> 
> ```Plain
> [Instruction Prompt] +
> [Retrieved Context (kNN hoặc S3)] +
> [Current Dialogue Turn]
> → Input vào LLM (GPT/PaLM) → Sinh phản hồi
> ```
> 
> Instruction prompt là phần hướng dẫn LLM cách sử dụng ngữ cảnh đã truy xuất để trả lời.
> 
> ---
> 
> #### 🧪 **3. Đánh giá**
> 
> - Được thực nghiệm trên 2 tập:
>     
>     - **Multi-Session Chat (MSC)**: hội thoại nhiều phiên, đánh giá khả năng “nhớ” persona.
>         
>     - **MultiDoc2Dial**: hội thoại tri thức, truy xuất đa tài liệu.
>         
> - Chỉ số đo: BLEURT, ROUGE-L, METEOR, F1-persona
>     
> - Kết quả: **Truy xuất bằng kNN hoặc S3 vượt cả tóm tắt thủ công**, hiệu quả cao mà không cần fine-tune.
>     
> 
> ---
> 
> ### ✅ Tóm gọn cho mục (3):
> 
> Bài báo _Retrieval-Augmented Generation for Dialog Modeling_ đề xuất một baseline đơn giản kết hợp truy xuất đoạn hội thoại cũ (bằng kNN hoặc tóm tắt submodular – S3) và ghép vào prompt để cung cấp ngữ cảnh cho mô hình LLM. Hệ thống này không cần fine-tune mô hình, nhưng vẫn đạt hiệu quả cao trên các tập hội thoại nhiều phiên, cho thấy sức mạnh của nén ngữ cảnh và truy xuất đúng đoạn thông tin cần thiết trong việc cải thiện trí nhớ dài hạn của hệ thống đối thoại.
> 
> ---

  

Phương pháp **Nén ngữ cảnh và Truy xuất khi cần** đã được áp dụng trong nhiều nghiên cứu khác nhau nhằm cải thiện khả năng ghi nhớ dài hạn của mô hình hội thoại. Dưới đây là một số nghiên cứu tiêu biểu:

1. **"Retrieval-Augmented Generation for Dialog Modeling" (NeurIPS 2023):**
    
    1. **Tóm tắt:** Nghiên cứu này đề xuất việc sử dụng mô hình ngôn ngữ lớn (LLM) kết hợp với truy xuất thông tin để cải thiện khả năng tạo phản hồi trong hội thoại dài hạn. Phương pháp này không yêu cầu tinh chỉnh mô hình và tập trung vào việc truy xuất thông tin liên quan từ ngữ cảnh hội thoại trước đó hoặc từ các nguồn tri thức bên ngoài.
        
    2. **Liên kết:** [https://neurips.cc/virtual/2023/81162](https://neurips.cc/virtual/2023/81162)
        
2. **"Learning Retrieval Augmentation for Personalized Dialogue Generation" (EMNLP 2023):**
    
    1. **Tóm tắt:** Bài báo này giới thiệu mô hình LAPDOG, sử dụng truy xuất thông tin từ các câu chuyện liên quan để bổ sung ngữ cảnh cho hồ sơ cá nhân, nhằm tạo ra các phản hồi hội thoại được cá nhân hóa hơn.
        
    2. **Liên kết:** [https://aclanthology.org/2023.emnlp-main.154/](https://aclanthology.org/2023.emnlp-main.154/)
        
3. **"Retrieval-Augmented Generation for Large Language Models: A Survey" (arXiv 2023):**
    
    1. **Tóm tắt:** Bài khảo sát này tổng hợp các phương pháp kết hợp truy xuất thông tin với mô hình ngôn ngữ lớn, nhấn mạnh việc sử dụng dữ liệu bên ngoài để cải thiện độ chính xác và tính cập nhật của phản hồi.
        
    2. **Liên kết:** [https://arxiv.org/abs/2312.10997](https://arxiv.org/abs/2312.10997)
        
4. **"Self-RAG: Self-reflective Retrieval Augmented Generation" (NeurIPS 2023):**
    
    1. **Tóm tắt:** Nghiên cứu này giới thiệu khung làm việc Self-RAG, cho phép mô hình tự động truy xuất và phản ánh thông tin trong quá trình tạo phản hồi, nhằm nâng cao chất lượng và tính chính xác của phản hồi.
        
    2. **Liên kết:** [https://neurips.cc/virtual/2023/79625](https://neurips.cc/virtual/2023/79625)
        
5. **"Retrieval-Augmented Neural Response Generation Using Logical Reasoning and Relevance Scoring" (arXiv 2023):**
    
    1. **Tóm tắt:** Bài báo này đề xuất kết hợp mô hình ngôn ngữ với suy luận logic và đánh giá mức độ liên quan để cải thiện chất lượng phản hồi trong hệ thống hội thoại.
        
    2. **Liên kết:** [https://arxiv.org/abs/2310.13566](https://arxiv.org/abs/2310.13566)
        

Những nghiên cứu này cho thấy sự phát triển và ứng dụng đa dạng của phương pháp Nén ngữ cảnh và Truy xuất khi cần trong việc cải thiện hiệu suất của các mô hình hội thoại dài hạn.