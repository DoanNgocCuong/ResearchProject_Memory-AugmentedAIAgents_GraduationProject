TEST_DATA = [
    {
        "query": "Khi nào tôi sẽ nhận được Robot Pika nếu đặt mua pre-order?",
        "expected_answer_id": 0,
        "expected_answer": "Robot dự kiến hoàn thiện trong 4 tháng (+ tối đa 2 tháng chênh lệch). Nếu sau 4 tháng vẫn chưa giao, bạn có quyền yêu cầu hoàn tiền."
    },
    {
        "query": "Thời gian giao hàng tối đa của Pre-order là bao lâu?",
        "expected_answer_id": 0,
        "expected_answer": "Robot dự kiến hoàn thiện trong 4 tháng (+ tối đa 2 tháng chênh lệch). Nếu sau 4 tháng vẫn chưa giao, bạn có quyền yêu cầu hoàn tiền."
    },
    {
        "query": "Tôi cần đặt cọc bao nhiêu khi đăng ký Pre-order?",
        "expected_answer_id": 1,
        "expected_answer": "Tùy vào đợt Pre-order, bạn có thể cần đặt cọc một khoản cố định. Thông tin cụ thể sẽ có trong hướng dẫn thanh toán."
    },
    {
        "query": "Robot Pika phù hợp cho độ tuổi và trình độ tiếng Anh nào?",
        "expected_answer_id": 2,
        "expected_answer": "Phiên bản đầu tiên tối ưu cho trẻ ở trình độ A1–A1.5 (tương đương Movers – Flyers). Trình độ phù hợp nhất sẽ nằm trong khoảng từ 5 - 10 tuổi."
    },
    {
        "query": "Nhà tôi có hai bé, có cần mua hai robot riêng không?",
        "expected_answer_id": 3,
        "expected_answer": "Phiên bản hiện tại tối ưu cho 1 người dùng chính. Nếu cần cho 2 bé, có thể có phụ phí hoặc tuỳ chọn nâng cấp trong tương lai."
    },
    {
        "query": "Sau năm đầu, phí subscription của Robot là bao nhiêu?",
        "expected_answer_id": 4,
        "expected_answer": "Năm đầu tiên đã bao gồm trong giá bán (gói 2.250k). Từ năm thứ hai, phí sẽ là 599k/năm nếu muốn duy trì toàn bộ tính năng."
    },
    {
        "query": "Robot được bảo hành như thế nào?",
        "expected_answer_id": 5,
        "expected_answer": "Bảo hành 1 năm cho lỗi do nhà sản xuất. Các lỗi do người dùng sẽ tính phí sửa chữa."
    },
    {
        "query": "Việc phụ huynh hỗ trợ con sử dụng Robot có phức tạp không?",
        "expected_answer_id": 6,
        "expected_answer": "Robot giao tiếp với bé, phụ huynh chỉ cần cài app và cài đặt ban đầu. Có hỗ trợ thêm nếu cần."
    },
    {
        "query": "Tôi muốn huỷ đơn Pre-order có được hoàn cọc không?",
        "expected_answer_id": 7,
        "expected_answer": "Có thể huỷ nếu Robot chưa vào sản xuất, hoàn cọc theo chính sách công ty."
    },
    {
        "query": "Nội dung học của Pika có linh hoạt tuỳ sở thích của bé không?",
        "expected_answer_id": 8,
        "expected_answer": "Có thể tuỳ chỉnh nội dung theo sở thích bé."
    },
    {
        "query": "Sau khi nhận Robot, tôi phải cài đặt như thế nào?",
        "expected_answer_id": 9,
        "expected_answer": "Kết nối với Wi-Fi và app điện thoại, Robot sẽ tự động cập nhật."
    },
    {
        "query": "Robot sửa phát âm tiếng Anh của bé bằng cách nào?",
        "expected_answer_id": 10,
        "expected_answer": "Pika dùng nhận diện giọng nói để sửa lỗi phát âm và hướng dẫn luyện tập."
    },
    {
        "query": "Tôi có thể liên hệ hỗ trợ kênh nào nếu có thắc mắc về Robot?",
        "expected_answer_id": 11,
        "expected_answer": "Qua hotline, email hoặc Zalo để được hỗ trợ."
    },
    {
        "query": "Robot có dùng được khi offline không?",
        "expected_answer_id": 12,
        "expected_answer": "Một số chức năng cơ bản hoạt động offline, nhưng cần Wi-Fi để cập nhật nội dung."
    },
    {
        "query": "Nếu Robot báo lỗi phần mềm, tôi cần làm gì?",
        "expected_answer_id": 13,
        "expected_answer": "Robot hỗ trợ cập nhật OTA và kỹ thuật viên hỗ trợ khi cần."
    },
    {
        "query": "Pika có giúp bé phát triển kỹ năng nào ngoài tiếng Anh không?",
        "expected_answer_id": 14,
        "expected_answer": "Có hỗ trợ kỹ năng giao tiếp, EQ và tư duy sáng tạo."
    },
    {
        "query": "Nếu làm rơi Robot hỏng, có thể thay linh kiện ở đâu?",
        "expected_answer_id": 15,
        "expected_answer": "Có thể liên hệ trung tâm hỗ trợ để sửa chữa hoặc thay linh kiện."
    },
    {
        "query": "Có phụ kiện nào kèm theo khi mua Robot không?",
        "expected_answer_id": 16,
        "expected_answer": "Robot đã kèm phụ kiện cơ bản, phụ kiện nâng cấp có bán riêng."
    },
    {
        "query": "Tôi có thể giới hạn thời gian bé chơi Robot mỗi ngày không?",
        "expected_answer_id": 17,
        "expected_answer": "Phụ huynh có thể cài đặt giới hạn qua app."
    },
    {
        "query": "Robot có được cập nhật bài học mới thường xuyên không?",
        "expected_answer_id": 18,
        "expected_answer": "Có, nội dung được cập nhật mỗi tháng nếu còn thời gian subscription."
    },
    {
        "query": "Khi bé học xong cấp độ hiện tại, có gói nâng cấp cao hơn không?",
        "expected_answer_id": 19,
        "expected_answer": "Sẽ có các gói mở rộng A2, B1 trong tương lai."
    },
    {
        "query": "Robot có thể cá nhân hoá bài học cho từng bé không?",
        "expected_answer_id": 20,
        "expected_answer": "Có, nội dung được điều chỉnh theo sở thích và tốc độ học của bé."
    },
    {
        "query": "Robot hỗ trợ nhiều tài khoản người dùng không?",
        "expected_answer_id": 21,
        "expected_answer": "Hiện ưu tiên 1 tài khoản chính, nhưng app quản lý nhiều bé, sẽ nâng cấp thêm profile trong tương lai."
    },
    {
        "query": "Điều gì xảy ra nếu tôi không gia hạn subscription?",
        "expected_answer_id": 4,
        "expected_answer": "Năm đầu tiên đã bao gồm trong giá bán (gói 2.250k). Từ năm thứ hai, phí sẽ là 599k/năm nếu muốn duy trì toàn bộ tính năng."
    },
    {
        "query": "Robot sẽ hoạt động ra sao khi mất kết nối Wi-Fi?",
        "expected_answer_id": 12,
        "expected_answer": "Một số chức năng cơ bản hoạt động offline, nhưng cần Wi-Fi để cập nhật nội dung."
    },
    {
        "query": "Bảo hành có áp dụng nếu Robot bị nước vào do tôi bất cẩn không?",
        "expected_answer_id": 5,
        "expected_answer": "Bảo hành 1 năm cho lỗi do nhà sản xuất. Các lỗi do người dùng sẽ tính phí sửa chữa."
    },
    {
        "query": "Trong hộp Robot Pika gồm những phụ kiện gì?",
        "expected_answer_id": 16,
        "expected_answer": "Robot đã kèm phụ kiện cơ bản, phụ kiện nâng cấp có bán riêng."
    },
    {
        "query": "Tôi muốn đặt giới hạn 30 phút sử dụng mỗi ngày, làm thế nào?",
        "expected_answer_id": 17,
        "expected_answer": "Phụ huynh có thể cài đặt giới hạn qua app."
    },
    {
        "query": "Pika thúc đẩy EQ của trẻ qua hoạt động nào?",
        "expected_answer_id": 14,
        "expected_answer": "Có hỗ trợ kỹ năng giao tiếp, EQ và tư duy sáng tạo."
    },
    {
        "query": "Làm sao Robot điều chỉnh nội dung theo sở thích của bé?",
        "expected_answer_id": 20,
        "expected_answer": "Có, nội dung được điều chỉnh theo sở thích và tốc độ học của bé."
    }
] 