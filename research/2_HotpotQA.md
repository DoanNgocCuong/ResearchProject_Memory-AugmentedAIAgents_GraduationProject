
## 📦 **Cấu trúc cấp cao (Top-level Structure)**

Toàn bộ file JSON là một **danh sách (list)**, trong đó **mỗi phần tử là một điểm dữ liệu (data point)**, tương ứng với **1 câu hỏi và câu trả lời (QA pair)**.

---

## 🔑 **Chi tiết các khóa trong mỗi data point (dict)**

### 1. **`_id`**:

* Kiểu: `string`
* Ví dụ: `"5a7a06935542990198eaf050"`
* Chức năng: Mã định danh duy nhất cho từng QA item, dùng để theo dõi hoặc đánh giá.

---

### 2. **`question`**:

* Kiểu: `string`
* Ví dụ: `"Which magazine was started first Arthur's Magazine or First for Women?"`
* Câu hỏi mà hệ thống cần trả lời.

---

### 3. **`answer`** *(không có trong test set)*:

* Kiểu: `string`
* Ví dụ: `"Arthur's Magazine"`
* Câu trả lời ngắn gọn (factoid) tương ứng với câu hỏi.

---

### 4. **`supporting_facts`** *(không có trong test set)*:

* Kiểu: `list`
* Ví dụ:

```json
[
  ["Arthur's Magazine", 0],
  ["First for Women", 0]
]
```

* Giải thích:

  * Danh sách các đoạn văn và vị trí câu (sent\_id) **chứa thông tin hỗ trợ việc trả lời câu hỏi**.
  * Mỗi phần tử là một cặp `[title, sent_id]`:

    * `title`: tiêu đề của đoạn văn.
    * `sent_id`: chỉ số câu trong đoạn đó (đếm từ 0).

---

### 5. **`context`**:

* Kiểu: `list`
* Mỗi phần tử là:

```json
[
  "Title of paragraph",
  ["Sentence 1", "Sentence 2", ..., "Sentence n"]
]
```

* Giải thích:

  * Mỗi phần tử đại diện cho một đoạn văn (paragraph).
  * Bao gồm:

    * **Tiêu đề đoạn (`title`)** – thường là tiêu đề trang Wikipedia.
    * **Danh sách câu (`sentences`)** – nội dung đoạn văn, tách thành từng câu.

---

## 🔄 **Các key bổ sung (không dùng trong test)**

### 6. **`type`** *(optional)*:

* Kiểu: `"comparison"` hoặc `"bridge"`
* Ý nghĩa:

  * `comparison`: Câu hỏi yêu cầu so sánh giữa 2 facts (VD: A hay B đến trước?)
  * `bridge`: Câu hỏi cần nối 2 thông tin lại với nhau để trả lời.

---

### 7. **`level`** *(optional)*:

* Kiểu: `"easy"`, `"medium"`, `"hard"`
* Chỉ độ khó tương đối của câu hỏi.

---

## ✨ **Ví dụ đầy đủ**

```json
{
  "_id": "5a7a06935542990198eaf050",
  "question": "Which magazine was started first Arthur's Magazine or First for Women?",
  "answer": "Arthur's Magazine",
  "type": "comparison",
  "level": "easy",
  "supporting_facts": [
    ["Arthur's Magazine", 0],
    ["First for Women", 0]
  ],
  "context": [
    [
      "Arthur's Magazine",
      [
        "Arthur's Magazine was an American literary periodical published in the 1840s.",
        "It was founded by Timothy Shay Arthur in Philadelphia."
      ]
    ],
    [
      "First for Women",
      [
        "First for Women is a women's magazine published by Bauer Media Group.",
        "It was launched in 1989."
      ]
    ]
  ]
}
```

---

## ✅ Tóm tắt nhanh

| Trường             | Kiểu dữ liệu | Mục đích                                 |
| ------------------ | ------------ | ---------------------------------------- |
| `_id`              | string       | ID duy nhất cho QA                       |
| `question`         | string       | Câu hỏi                                  |
| `answer`           | string       | Câu trả lời đúng                         |
| `supporting_facts` | list         | Vị trí các câu quan trọng hỗ trợ trả lời |
| `context`          | list         | Danh sách đoạn văn có tiêu đề + câu      |
| `type`             | string       | Loại câu hỏi (bridge/comparison)         |
| `level`            | string       | Mức độ khó                               |



---

```bash
{
  "_id": "5a7a06935542990198eaf050",
  "question": "Which magazine was started first Arthur's Magazine or First for Women?",
  "answer": "Arthur's Magazine",
  "type": "comparison",
  "level": "easy",
  "supporting_facts": [
    ["Arthur's Magazine", 0],
    ["First for Women", 0]
  ],
  "context": [
    [
      "Arthur's Magazine",
      [
        "Arthur's Magazine was an American literary periodical published in the 1840s.",
        "It was founded by Timothy Shay Arthur in Philadelphia."
      ]
    ],
    [
      "First for Women",
      [
        "First for Women is a women's magazine published by Bauer Media Group.",
        "It was launched in 1989."
      ]
    ]
  ]
}

```

Convert sang 

