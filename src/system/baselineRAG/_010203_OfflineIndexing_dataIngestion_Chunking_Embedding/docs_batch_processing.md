Tôi sẽ tạo một file mới để xử lý việc gửi dữ liệu theo batch với retry logic:



Tôi đã tạo file `batch_processing.py` với các tính năng:

1. Xử lý theo batch:
   - Mặc định 100 documents/batch
   - Nghỉ 1 phút giữa các batch
   - Có thể tùy chỉnh batch size và thời gian nghỉ

2. Retry logic:
   - Thử lại tối đa 3 lần khi gặp lỗi
   - Thời gian chờ tăng dần: 5, 10, 15 phút
   - Log chi tiết mỗi lần retry

3. Logging chi tiết:
   - Log ra file trong thư mục `logs`
   - Tên file log có timestamp
   - Log cả ra console và file
   - Format: `timestamp - level - message`

4. Command line arguments:
```bash
python batch_processing.py --excel data.xlsx --start 0 --end 1000 --batch-size 100 --wait-time 60
```

Ví dụ log output:
```
2024-03-14 10:00:00 - INFO - Starting batch processing:
2024-03-14 10:00:00 - INFO - - Total documents to process: 1000
2024-03-14 10:00:00 - INFO - - Batch size: 100
2024-03-14 10:00:00 - INFO - - Number of batches: 10
2024-03-14 10:00:00 - INFO - - Wait time between batches: 60 seconds

2024-03-14 10:00:00 - INFO - Processing batch 1/10
2024-03-14 10:00:00 - INFO - Documents 0 to 99
...
```

Bạn có thể chạy thử với:
```bash
python batch_processing.py --excel src/datasets/dataset_full/vimqa_processed/corpus_vimqa_dev_300.xlsx --start 0 --end 10
```
