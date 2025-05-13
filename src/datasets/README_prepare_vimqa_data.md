# Hướng dẫn sử dụng prepare_vimqa_data.py và package vimqa

## 1. Mục đích

Script `prepare_vimqa_data.py` và package `vimqa` giúp bạn chuyển đổi dữ liệu VIMQA (hoặc HotpotQA dạng tương tự) thành các file nhỏ phục vụ cho các hệ thống truy hồi và trả lời câu hỏi (retrieval-based QA) như RAG, FiD, DPR, ColBERT...

Các file đầu ra gồm:
- **corpus_{input}.jsonl**: Kho tri thức (corpus) để retriever truy hồi
- **qa_pairs_{input}.jsonl**: Danh sách câu hỏi, đáp án, supporting facts
- **contexts_gold_{input}.jsonl**: Contexts đã gắn nhãn đúng/sai để train reader
- **{input}_vi.json**: File JSON tiếng Việt dễ đọc (không mã hóa unicode escape)

---

## 2. Cách sử dụng

### a. Cài đặt thư mục

Đảm bảo cấu trúc thư mục như sau:
```
src/datasets/
├── prepare_vimqa_data.py
└── vimqa/
    ├── __init__.py
    ├── corpus.py
    ├── qa_pairs.py
    ├── contexts.py
    └── utils.py
```

### b. Chạy script

Giả sử bạn có file `train.json`:
```bash
python prepare_vimqa_data.py train.json data/vimqa_processed
```
- Tham số 1: Đường dẫn file input (VD: `train.json`)
- Tham số 2: Thư mục output (VD: `data/vimqa_processed`)

Sau khi chạy, bạn sẽ có các file:
- `corpus_train.jsonl`
- `qa_pairs_train.jsonl`
- `contexts_gold_train.jsonl`
- `train_vi.json`

---

## 3. Ý nghĩa các file đầu ra

### corpus_{input}.jsonl
- Mỗi dòng là 1 đoạn văn (document) từ context
- Dùng để build index cho retriever (BM25, FAISS, ColBERT...)
- Ví dụ:
```json
{"doc_id": "Donald Trump_0", "title": "Donald Trump", "text": "Donald Trump là..."}
```

### qa_pairs_{input}.jsonl
- Mỗi dòng là 1 cặp câu hỏi, đáp án, supporting facts
- Dùng để train reader hoặc đánh giá EM/F1
- Ví dụ:
```json
{"question_id": "abc123", "question": "Donald Trump là ai?", "answer": "Tổng thống Mỹ", "supporting_facts": [["Donald Trump", 0]]}
```

### contexts_gold_{input}.jsonl
- Mỗi dòng là 1 câu hỏi, đáp án, và danh sách context đã gắn nhãn đúng/sai
- Dùng để train reader hoặc đánh giá joint EM/F1
- Ví dụ:
```json
{
  "question_id": "abc123",
  "question": "Donald Trump là ai?",
  "answer": "Tổng thống Mỹ",
  "contexts": [
    {"title": "Donald Trump", "text": "Donald Trump là...", "is_supporting": true},
    {"title": "Barack Obama", "text": "Barack Obama là...", "is_supporting": false}
  ]
}
```

### {input}_vi.json
- Bản tiếng Việt dễ đọc của file input (không mã hóa unicode escape)
- Dùng để kiểm tra dữ liệu hoặc debug

---

## 4. Giải thích chi tiết code

### a. File prepare_vimqa_data.py

- **Chức năng:** Script chính, gọi các hàm xử lý từ package `vimqa` để tạo các file output.
- **Các bước chính:**
  1. Đọc dữ liệu từ file JSON input
  2. Tạo và lưu các file corpus, qa_pairs, contexts_gold
  3. Tạo file tiếng Việt dễ đọc
- **Các hàm sử dụng:**
  - `create_corpus`: Tạo danh sách các đoạn văn từ context
  - `create_qa_pairs`: Tạo danh sách câu hỏi, đáp án, supporting facts
  - `create_contexts_gold`: Tạo danh sách context đã gắn nhãn
  - `save_jsonl`: Lưu dữ liệu ra file JSONL
  - `convert_to_vietnamese_readable`: Lưu file JSON tiếng Việt dễ đọc

#### Ví dụ đoạn code chính:
```python
import argparse
from vimqa import (
    create_corpus,
    create_qa_pairs,
    create_contexts_gold,
    save_jsonl,
    load_json,
    convert_to_vietnamese_readable
)
import os

def process_vimqa_file(input_file: str, output_dir: str):
    data = load_json(input_file)
    input_base = os.path.splitext(os.path.basename(input_file))[0]
    save_jsonl(create_corpus(data), f"{output_dir}/corpus_{input_base}.jsonl")
    save_jsonl(create_qa_pairs(data), f"{output_dir}/qa_pairs_{input_base}.jsonl")
    save_jsonl(create_contexts_gold(data), f"{output_dir}/contexts_gold_{input_base}.jsonl")
    vi_json_path = os.path.join(output_dir, f"{input_base}_vi.json")
    convert_to_vietnamese_readable(input_file, vi_json_path)
```

### b. Package vimqa

#### 1. corpus.py
- Hàm `create_corpus(data)`
  - Duyệt qua từng context, tạo doc_id duy nhất cho mỗi đoạn văn
  - Trả về list các document dạng dict

#### 2. qa_pairs.py
- Hàm `create_qa_pairs(data)`
  - Lấy các trường _id, question, answer, supporting_facts từ mỗi example
  - Trả về list các cặp QA

#### 3. contexts.py
- Hàm `create_contexts_gold(data)`
  - Duyệt qua từng câu trong context, gắn nhãn is_supporting dựa vào supporting_facts
  - Trả về list các context đã gắn nhãn

#### 4. utils.py
- Hàm `save_jsonl(data, output_file)`
  - Lưu list dict ra file JSONL
- Hàm `load_json(input_file)`
  - Đọc file JSON vào list dict
- Hàm `convert_to_vietnamese_readable(input_file, output_file)`
  - Lưu lại file JSON với tiếng Việt dễ đọc (ensure_ascii=False)

#### 5. __init__.py
- Export các hàm chính để import dễ dàng từ package

---

## 5. Comment chi tiết trong code

Tất cả các hàm đều có docstring giải thích rõ:
- Mục đích hàm
- Tham số vào/ra
- Ý nghĩa các trường dữ liệu

Bạn có thể đọc trực tiếp trong từng file `.py` để xem chi tiết.

---

## 6. Liên hệ & đóng góp

Nếu bạn có câu hỏi hoặc muốn đóng góp thêm, hãy tạo issue hoặc pull request trên repo!

---

**Chúc bạn thành công với pipeline QA!**


---


Câu hỏi rất hay! Nếu bạn muốn triển khai hoặc huấn luyện hệ thống **retrieval-based QA** như RAG, FiD, DPR, ColBERT, hay QA inference theo kiểu "truy từ corpus", thì bạn **thực sự cần tách dữ liệu HotpotQA ra thành nhiều file riêng biệt**. Dưới đây là **bộ file tối thiểu nên chuẩn bị**, kèm theo **mục đích rõ ràng cho từng file**.

---

## ✅ 1. `corpus.jsonl`  – **Kho tri thức / cơ sở dữ liệu để truy hồi**

### 📦 Nội dung:

Danh sách tất cả các đoạn văn (documents), mỗi dòng là một JSON object:

```json
{"doc_id": "Donald Trump", "title": "Donald Trump", "text": "Là chủ tịch kiêm... bản tính thẳng thắn của mình."}
{"doc_id": "Mike Pence", "title": "Mike Pence", "text": "Michael Richard 'Mike' Pence..."}
...
```

### 🧠 Mục đích:

* Làm **index corpus** cho retriever (BM25, FAISS, ColBERT…)
* Cho retriever "đi tìm tài liệu" khi nhận câu hỏi
* Có thể build FAISS index từ đây

---

## ✅ 2. `qa_pairs.jsonl` – **Danh sách các câu hỏi + đáp án + supporting\_facts**

### 📦 Nội dung:

```json
{
  "question_id": "3d19e0cf-...",
  "question": "Donald Trump nổi tiếng toàn nước Mỹ nhờ gì",
  "answer": "nhờ sự nghiệp...",
  "supporting_facts": [["Donald Trump", 1]]
}
```

### 🧠 Mục đích:

* Huấn luyện reader (trả lời dựa trên context)
* Đánh giá EM/F1
* Nếu bạn dùng open QA: `question` → `retriever` → `reader`

---

## ✅ 3. `contexts_gold.jsonl` – **Dữ liệu context thực tế dùng để huấn luyện reader**

### 📦 Nội dung:

```json
{
  "question_id": "3d19e0cf-...",
  "question": "...",
  "answer": "...",
  "contexts": [
    {
      "title": "Donald Trump",
      "text": "Là chủ tịch kiêm...",
      "is_supporting": true
    },
    {
      "title": "Charles J. Pedersen",
      "text": "Charles John Pedersen là nhà hóa học...",
      "is_supporting": false
    }
  ]
}
```

### 🧠 Mục đích:

* Để huấn luyện hoặc fine-tune reader trên dữ liệu đã biết context
* Dùng cho evaluation (Joint EM/F1)

---

## ✅ Tóm gọn – Bạn nên tạo 3–4 file như sau:

| Tên file                    | Mục đích chính                             | Bắt buộc?                       |
| --------------------------- | ------------------------------------------ | ------------------------------- |
| ✅ `corpus.jsonl`            | Tập đoạn văn gốc để truy hồi               | **Bắt buộc**                    |
| ✅ `qa_pairs.jsonl`          | Tập câu hỏi + đáp án                       | **Bắt buộc**                    |
| ✅ `contexts_gold.jsonl`     | Contexts gắn nhãn đúng/sai để train reader | **Rất nên**                     |

---

## 🧪 Ví dụ pipeline:

1. `question` → tìm trong `corpus.jsonl` → top-k docs
2. → đưa docs vào `reader` → ra `answer`
3. → so với `qa_pairs.jsonl["answer"]` để tính EM/F1

---
