Lỗi này xảy ra vì Qdrant cần có range index cho field `id` để sử dụng `order_by`. Đây là các cách khác để lấy 3 points cuối cùng:

## 1. **Cách đơn giản nhất - Scroll ngược từ cuối:**

```http
POST collections/VIMQA_dev/points/scroll
{
  "limit": 3,
  "with_payload": true,
  "with_vectors": true
}
```

Sau đó từ kết quả, bạn lấy 3 elements cuối cùng.

## 2. **Đếm total points trước, rồi dùng offset:**

Đầu tiên đếm:
```http
POST collections/VIMQA_dev/points/count
{
  "exact": true
}
```

Giả sử có 10 points, thì lấy 3 cuối:
```http
POST collections/VIMQA_dev/points/scroll
{
  "limit": 3,
  "offset": 7,
  "with_payload": true,
  "with_vectors": true
}
```

## 3. **Tạo range index cho field id (nếu muốn dùng order_by):**

```http
PUT collections/VIMQA_dev/index
{
  "field_name": "id",
  "field_schema": "integer"
}
```

Hoặc nếu id là string:
```http
PUT collections/VIMQA_dev/index
{
  "field_name": "id", 
  "field_schema": "keyword"
}
```

## 4. **Sử dụng field khác có thể sort được:**

Nếu trong payload có trường timestamp hoặc created_at:
```http
POST collections/VIMQA_dev/points/scroll
{
  "limit": 3,
  "order_by": {
    "key": "timestamp",
    "direction": "desc"
  },
  "with_payload": true,
  "with_vectors": true
}
```

## 5. **Cách thực tế nhất - Lấy tất cả rồi slice:**

```http
POST collections/VIMQA_dev/points/scroll
{
  "limit": 100,
  "with_payload": true,
  "with_vectors": false
}
```

Từ response, bạn lấy 3 elements cuối cùng trong array `points`.

## 6. **Kiểm tra cấu trúc collection trước:**

```http
GET collections/VIMQA_dev
```

Để xem collection có bao nhiêu points và cấu trúc như thế nào.

**Recommendation:** Dùng cách số 2 (count + offset) là đáng tin cậy nhất và không cần tạo thêm index.
