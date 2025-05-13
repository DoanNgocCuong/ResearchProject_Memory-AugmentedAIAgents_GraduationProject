# Đề cương đồ án (draft v0.2)

> **Lưu ý**: Đây là sườn nội dung theo template ĐATN SOICT và đã cập nhật (1) tuyên bố vấn đề thiếu hụt của các giải pháp RAG hiện nay, (2) mô tả dataset sử dụng trong đồ án (theo Nguyễn Đăng Huy, 2025), (3) kế hoạch thực nghiệm HippoRAG 2.

---

## Chương 1 Giới thiệu

### 1.1 Đặt vấn đề

Các hệ thống **Retrieval‑Augmented Generation (RAG)** đang được sử dụng rộng rãi để khắc phục hiện tượng *hallucination* của LLM. Tuy nhiên, phần lớn giải pháp hiện hành vẫn **dựa hoàn toàn vào truy xuất vector** nên:

1. Khó **tích hợp tri thức phân tán** (multi‑hop associativity).
2. Thiếu khả năng **sense‑making** cho ngữ cảnh dài, phức tạp.
3. Độ linh hoạt kém khi **cập nhật dữ liệu liên tục** (continual learning).

### 1.2 Mục tiêu

* Khảo sát & phân tích hạn chế của RAG truyền thống.
* Triển khai **HippoRAG 2** – khung RAG sử dụng Personalized PageRank trên **kiến thức đồ thị mở (open KG)** – áp dụng cho **ngôn ngữ mục tiêu (Việt/Anh, sẽ chốt sau)**.
* Đánh giá trên bộ dữ liệu mở rộng từ HotpotQA (Nguyễn Đăng Huy, 2025) và so sánh với các baseline phổ biến (BM25, Contriever, NV‑Embed, RAPTOR,…).

### 1.3 Đóng góp dự kiến

1. Áp dụng thành công HippoRAG 2 cho **bộ dữ liệu lựa chọn (Việt hoặc Anh)**, cung cấp mã nguồn & dữ liệu tiền xử lý.
2. Phân tích định lượng ảnh hưởng của **kỹ thuật gán trọng số seed node** (phrase node vs. passage node) và hệ số w trong quá trình PageRank.
3. Báo cáo khuyến nghị khi triển khai HippoRAG 2 cho các bài toán doanh nghiệp tại Việt Nam.

---

## Chương 2 Nền tảng lý thuyết (≤10 trang)

* Tóm lược mô hình ngôn ngữ lớn và hiện tượng hallucination.
* Kiến trúc RAG chuẩn & các biến thể: vector‑RAG, GraphRAG, RAPTOR,…
* **HippoRAG v1 & v2**:

  * Cấu trúc 3 thành phần: *neocortex* (LLM), *parahippocampal region* (retriever), *hippocampus* (open KG)citeturn9file11.
  * Cải tiến v2: **dense‑sparse integration & recognition memory** nhằm chọn seed node tốt hơn và giảm nhiễuciteturn9file1.

---

## Chương 3 Phương pháp đề xuất

### 3.1 Tổng quan pipeline

1. **Offline indexing**: OpenIE (Llama‑3‑70B‑Instruct) sinh triple, thêm synonym‑edge; gắn phrase với passage node để tạo KG kết hợp dense‑sparse.
2. **Seed initialisation**:

   * Chọn **top‑k phrase triple** theo cosine; lọc nhiễu bằng prompt *Recognition Memory* (DSPy tuned).
   * Chọn **top‑k passage**; trọng số seed = cosine × *w* (mặc định *w = 0.05* để “pha loãng” passage‑node)citeturn9file1.
3. **Personalised PageRank (α=0.5)** chạy trên KG → ranking passage.
4. **QA Reader**: Llama‑3‑70B‑Instruct.

### 3.2 Lan truyền trọng số (PageRank)

* Mỗi bước: *p<sub>t+1</sub> = (1‑α)·Mᵀ·p<sub>t</sub> + α·s*, với *M* ma trận chuyển tiếp chuẩn hoá theo bậc đỉnh; *s* là vector seed.
* **Phrase‑node** giữ ưu thế do tổng trọng số ban đầu cao hơn (passage seed đã nhân w).
* Sau \~20 iteration, xác suất hội tụ; cộng dồn lên passage node để lấy top‐N.

---

 Kết quả & thảo luận

* Bảng recall\@5 và EM/F1.
* Phân tích lý do HippoRAG 2 vượt trội (liên kết đa‐hop, giảm nhiễu).
* Thảo luận chi phí tính toán (LLM filter chỉ dùng cho top‑k, thời gian offline acceptable).

---

## Chương 6 Kết luận & hướng mở

* Tóm tắt đóng góp; hạn chế: KG chưa scale >1 triệu triple, tiếng Việt OpenIE còn lỗi.
* Định hướng: fine‑tune OpenIE tiếng Việt; thử Graph Neural Network thay PageRank.

---

## Lộ trình thực hiện (Gantt rút gọn)

| Tuần  | Công việc                                |
| ----- | ---------------------------------------- |
| 1‑2   | Nghiên cứu HippoRAG 2, thu thập mã nguồn |
| 3‑4   | Chuẩn hoá dataset HotpotQA‑VN‑Long       |
| 5‑7   | Xây dựng pipeline indexing & retrieval   |
| 8‑9   | Viết module filter (DSPy), chạy baseline |
| 10‑11 | Thực nghiệm chính, thống kê kết quả      |
| 12    | Hoàn thiện báo cáo & slide bảo vệ        |


