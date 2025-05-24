
```
src/datasets/
├── vimqa/
│   ├── __init__.py
│   ├── corpus.py      # Xử lý corpus
│   ├── qa_pairs.py    # Xử lý qa pairs
│   ├── contexts.py    # Xử lý contexts
│   └── utils.py       # Các hàm tiện ích
└── prepare_vimqa_data.py  # Script chính
```

Tôi đã refactor code thành các module nhỏ hơn với cấu trúc như sau:

1. **vimqa/corpus.py**: Xử lý việc tạo corpus
2. **vimqa/qa_pairs.py**: Xử lý việc tạo cặp câu hỏi-đáp án
3. **vimqa/contexts.py**: Xử lý việc tạo contexts với nhãn
4. **vimqa/utils.py**: Các hàm tiện ích (đọc/ghi file)
5. **vimqa/__init__.py**: Export các hàm cần thiết
6. **prepare_vimqa_data.py**: Script chính để chạy

Lợi ích của việc refactor này:

1. **Dễ bảo trì**: Mỗi module có một nhiệm vụ cụ thể
2. **Dễ mở rộng**: Có thể thêm tính năng mới vào từng module
3. **Dễ test**: Có thể viết test riêng cho từng module
4. **Tái sử dụng**: Các hàm tiện ích có thể dùng ở nhiều nơi
5. **Code sạch hơn**: Mỗi file ngắn gọn, dễ đọc

Để chạy script, bạn vẫn dùng lệnh như cũ:
```bash
python prepare_vimqa_data.py train.json data/vimqa_processed
```

Bạn có muốn tôi thêm các tính năng khác không? Ví dụ:
1. Thêm logging để theo dõi quá trình xử lý
2. Thêm validation cho dữ liệu đầu vào
3. Thêm unit tests cho các module
