# 1. HotpotQA Data Format Conversion

This document explains the data formats used in the HotpotQA dataset conversion process.

## Input Format (HotpotQA JSON)

The input data follows the HotpotQA dataset format, which is a JSON array containing question-answer pairs with supporting facts.

### Structure

```json
[
  {
    "_id": "unique_question_id",
    "question": "The question text",
    "supporting_facts": [
      ["title1", sentence_index1],
      ["title2", sentence_index2],
      ...
    ],
    "context": [
      ["title1", ["sentence1", "sentence2", ...]],
      ["title2", ["sentence1", "sentence2", ...]],
      ...
    ],
    "answer": "The answer text"
  }
]
```

Trùng với format của JSON Tiếng Việt trong: https://github.com/vimqa/vimqa/blob/main/dataset_examples/train.json


```bash
  {
    "_id": "5b68547e-f457-46fc-8d50-14c5507ec6bf",
    "question": "Tổ chức trao Giải Oscar được thành lập năm 1927 phải không?",
    "answer": "đúng",
    "context": [
      [
        "Viện Hàn lâm Khoa học và Nghệ thuật Điện ảnh",
        [
          "Viện Hàn lâm Khoa học và Nghệ thuật Điện ảnh (tiếng Anh: Academy of Motion Picture Arts and Sciences, thường viết tắt là AMPAS) là một tổ chức được thành lập năm 1927 tại Hoa Kỳ với mục đích vinh danh những thành tựu sáng tạo trong nghệ thuật điện ảnh.",
          "Hiện nay AMPAS có khoảng trên 6.000 hội viên là những người hoạt động trong lĩnh vực điện ảnh, phần lớn trong số họ là các nghệ sĩ điện ảnh Hoa Kỳ, ngoài ra trong thời gian gần đây AMPAS cũng kết nạp thêm các hội viên đến từ các nền điện ảnh khác, tính đến năm 2004 Viện có đại diện của 36 nền điện ảnh trên thế giới.AMPAS được biết đến nhiều nhất thông qua giải Oscar, giải thưởng điện ảnh lớn nhất của điện ảnh Hoa Kỳ.",
          "Giải Oscar được AMPAS bầu chọn và tổ chức từ năm 1929.",
          "Ngoài ra, AMPAS còn trao Giải Oscar sinh viên cho các nhà làm phim trẻ, một số học bổng chuyên ngành điện ảnh và quản lý Thư viện Margaret Herrick (Beverly Hills, California) cùng Trung tâm Nghiên cứu Điện ảnh Pickford (Hollywood, California)."
        ]
      ],
      [
        "Giải Oscar lần thứ 84",
        [
          "Giải Oscar lần thứ 84 được tổ chức bởi Viện Hàn lâm Khoa học và Nghệ thuật Điện ảnh Hoa Kỳ (Academy of Motion Picture Arts and Sciences - AMPAS) nhằm tôn vinh những phim xuất sắc nhất công chiếu trong năm 2011 diễn ra tại thuộc trung tâm thương mại Hollywood và Highland (trước kia gọi là nhà hát Kodak), thuộc khu Hollywood, thành phố Los Angeles, bang California, Hoa Kỳ bắt đầu từ lúc 5:30 chiều (Múi giờ Thái Bình Dương) hay 8:30 chiều (Múi giờ miền Đông Bắc Mỹ) ngày chủ nhật, 26/2/2012, tức 8:30 thứ hai, 27/2/2012 (giờ Hà Nội), được truyền hình trực tiếp tại Mỹ trên kênh ABC.Giải lần thứ 84 được trao cho 24 hạng mục, với các đề cử đã được Viện Hàn lâm công bố 2 ngày trước đó.",
          "Hai giải Oscar danh dự và một giải nhân đạo Jean Hersholt đã được trao ngày thứ bảy, 12 tháng 11 năm 2011.",
          "The Artist trở thành phim Pháp đầu tiên và là sự quay trở lại sau 83 năm vắng bóng của phim câm (sau Wings trong giải lần đầu tiên năm 1927) giành được giải chính.",
          "Cùng với Hugo của Martin Scorsese, The Artist của Michel Hazanavicius đã cùng chia sẻ ngôi vị phim đoạt nhiều giải thưởng nhất lần này: 5 giải."
        ]
      ],
      [
        "Giải vô địch bóng đá thế giới các câu lạc bộ",
        [
          "Cúp bóng đá thế giới các câu lạc bộ (tiếng Anh: FIFA Club World Cup), trước đây được gọi là FIFA Club World Championship (Giải vô địch bóng đá thế giới các câu lạc bộ), là một giải đấu bóng đá quốc tế dành cho nam được tổ chức bởi Liên đoàn bóng đá thế giới (FIFA), cơ quan quản lý bóng đá toàn cầu.",
          "Giải đấu lần đầu tiên được tổ chức với tên gọi Giải vô địch bóng đá thế giới các câu lạc bộ 2000.",
          "Giải sau đó không được tổ chức từ năm 2001 tới 2004 do nhiều nguyên nhân, nhưng quan trọng nhất vẫn là do sự sụp đổ của đối tác tiếp thị của FIFA là International Sport and Leisure.",
          "Từ năm 2005, giải được tổ chức hàng năm, và đã được tổ chức ở Brasil, Nhật Bản, Các Tiểu vương quốc Ả Rập Thống nhất và Maroc."
        ]
      ],
      [
        "Giải Grammy cho Album rock xuất sắc nhất",
        [
          "Giải Grammy cho Album rock xuất sắc nhất là một hạng mục trong lễ trao Giải Grammy, được thành lập vào năm 1958 và có tên gọi ban đầu là Giải Gramophone, được trao cho những nghệ sĩ có album thể loại nhạc rock hay nhất.",
          "Giải là một trong số các hạng mục luôn được Viện Hàn Lâm Nghệ thuật Thu Âm Hoa Kỳ trao tặng vào lễ trao giải thường niên nhằm \"tôn vinh các cá nhân/tập thể có thành tựu nghệ thuật xuất sắc trong lĩnh vực thu âm, mà không xét đến doanh số bán album hay vị trí trên các bảng xếp hạng âm nhạc\".",
          "Giải Grammy cho Album Rock xuất sắc nhất lần đầu được trao cho ban nhạc The Rolling Stones vào năm 1995 và không đổi tên giải cho đến ngày nay.",
          "Theo hướng dẫn mô tả về Giải Grammy lần thứ 52, giải thưởng này được trao cho \"album metal hay hard rock của ca sĩ hoặc trình diễn nhạc cụ Rock có thời lượng bản thu âm mới chiếm ít nhất 51%\".",
          "Kể từ năm 1996, đối tượng nhận giải ngoài ca sĩ trực tiếp thu âm thường bao gồm cả nhà sản xuất, người biên tập hoặc hòa âm phối khí cùng hợp tác tạo ra tác phẩm được đề cử."
        ]
      ],
      [
        "Giải thưởng Âm nhạc Melon",
        [
          "Giải thưởng Âm nhạc Melon (tiếng Anh: Melon Music Awards; tiếng Triều Tiên: 멜론 뮤직어워드) là một lễ trao giải âm nhạc lớn được tổ chức hàng năm tại Hàn Quốc và được tổ chức bởi Kakao M (một công ty của Kakao) thông qua cửa hàng âm nhạc trực tuyến, Melon.",
          "Lễ trao giải lần đầu tiên được tổ chức trực tuyến từ năm 2005 đến năm 2008.",
          "Nó được tổ chức trực tiếp chính thức bắt đầu từ năm 2009."
        ]
      ],
      [
        "Giải Oscar cho nam diễn viên chính xuất sắc nhất",
        [
          "Giải Oscar cho nam diễn viên chính xuất sắc nhất (tiếng Anh: Academy Award for Best Actor) là một hạng mục trong hệ thống giải Oscar do Viện Hàn lâm Khoa học và Nghệ thuật Điện ảnh (AMPAS) trao tặng hàng năm cho diễn viên nam có vai diễn chính xuất sắc nhất trong năm đó của ngành công nghiệp điện ảnh.",
          "Lễ trao giải Oscar lần thứ nhất được tổ chức vào năm 1929; Emil Jannings là chủ nhân đầu tiên của giải thưởng nhờ diễn xuất của ông trong các phim The Last Command và The Way of All Flesh.",
          "Ngày nay việc bầu chọn và trao đổi phiếu (do hội nam diễn viên trong Viện Hàn lâm thực hiện) có thể quyết định danh sách các ứng cử viên; còn số đông lá phiếu từ toàn bộ các thành viên có quyền biểu quyết hợp lệ của Viện Hàn lâm sẽ chọn ra những người chiến thắng.Trong ba năm đầu trao giải, các nam diễn viên giành đề cử cho màn thể hiện xuất sắc nhất trong các hạng mục của họ.",
          "Vào thời điểm đó toàn bộ tác phẩm của họ trong suốt giai đoạn vòng loại (trong một số trường hợp có tới ba phim) đều được đính tên trên giải thưởng.",
          "Tuy nhiên trong lễ trao giải lần thứ 3 tổ chức năm 1930, chỉ có một trong số những bộ phim đó được xướng tên trong giải thưởng cuối cùng của người thắng cử, mặc dù diễn xuất của mỗi người thắng cử có tới hai bộ phim đính tên họ trên những lá phiếu.",
          "Trong năm kế tiếp hệ thống hiện nay đã thay thế hệ thống phức tạp và khó sử dụng này, trong đó nam diễn viên đề cử cho một vai diễn cụ thể trong một bộ phim duy nhất.",
          "Kể từ lễ trao giải lần thứ 9, hạng mục được chính thức giới hạn năm đề cử mỗi năm.",
          "Kể từ khi thành lập, giải thưởng đã được trao cho 80 nam diễn viên.",
          "Daniel Day-Lewis giành nhiều giải nhất ở hạng mục này với ba tượng vàng.",
          "Spencer Tracy và Laurence Olivier đều nhận tới chín đề cử, nhiều hơn bất kì nam diễn viên nào khác."
        ]
      ],
      [
        "Giải Oscar",
        [
          "Giải thưởng Viện Hàn lâm (tiếng Anh: Academy Awards), thường được biết đến với tên Giải Oscar (tiếng Anh: Oscars) là giải thưởng điện ảnh hằng năm của Viện Hàn lâm Khoa học và Nghệ thuật Điện ảnh (tiếng Anh: Academy of Motion Picture Arts and Sciences, viết tắt là AMPAS) (Hoa Kỳ) với 74 giải thưởng dành cho các diễn viên và kỹ thuật hình ảnh trong ngành điện ảnh Hoa Kỳ.",
          "Kể từ năm 1929, giải Oscar được trao hàng năm tại thành phố Los Angeles để ghi nhận những thành tựu xuất sắc của điện ảnh trong năm của các đạo diễn, diễn viên, kịch bản và nhiều lĩnh vực khác qua cuộc bỏ phiếu kín của các thành viên Viện Hàn lâm."
        ]
      ],
      [
        "Giải Nobel",
        [
          "Giải thưởng Nobel, hay Giải Nobel (phát âm tiếng Thụy Điển: [noˈbɛl], Thụy Điển, số ít: Nobelpriset, Na Uy: Nobelprisen), là một tập các giải thưởng quốc tế được tổ chức trao thưởng hằng năm kể từ năm 1901 cho những cá nhân đạt thành tựu trong lĩnh vực vật lý, hoá học, y học, văn học, kinh tế và hòa bình; đặc biệt là giải hoà bình có thể được trao cho tổ chức hay cho cá nhân.",
          "Vào năm 1968, Ngân hàng Thụy Điển đưa thêm vào một giải về lĩnh vực khoa học kinh tế, theo di chúc của nhà phát minh người Thụy Điển Alfred Nobel năm 1895.",
          "Từ năm 1901 đến năm 2020, các giải thưởng Nobel và giải thưởng về Khoa học Kinh tế được trao tặng 603 lần cho 962 người và tổ chức.",
          "Do một số cá nhân và tổ chức nhận giải Nobel nhiều hơn một lần, tổng cộng có 962 cá nhân (905 nam và 57 nữ) và 25 tổ chức đã nhận giải này.Kết quả đoạt giải được công bố hằng năm vào tháng 10 và được trao (bao gồm tiền thưởng, một huy chương vàng và một giấy chứng nhận) vào ngày 10 tháng 12, ngày kỷ niệm ngày mất của Nobel.",
          "Giải Nobel được thừa nhận rộng rãi như là giải thưởng danh giá nhất một người có thể nhận được trong lĩnh vực được trao.Giải Nobel Hòa bình được trao thưởng ở Oslo, Na Uy, trong khi các giải khác được trao ở Stockholm, Thụy Điển.Viện Hàn lâm Khoa học Hoàng gia Thụy Điển trao giải Nobel Vật lý, giải Nobel Hóa học và giải Nobel Kinh tế; Hội Nobel ở Karolinska Institutet trao giải Nobel Sinh học và Y học; Viện Hàn lâm Thụy Điển trao giải Nobel Văn học; và giải thưởng Nobel Hòa bình được Ủy ban Nobel Na Uy (gồm 5 thành viên do Quốc hội Na Uy bầu ra ) trao tặng thay vì một tổ chức của Thụy Điển."
        ]
      ],
      [
        "Coppa Italia",
        [
          "Coppa Italia là một giải đấu cúp thường niên trong hệ thống các giải thi đấu nhà nghề của bóng đá Ý.",
          "Giải đấu đầu tiên được tổ chức vào năm 1922, nhưng giải lần thứ hai thì phải tới năm 1936 mới được tổ chức tiếp.",
          "Juventus là câu lạc bộ giàu thành tích nhất giải đấu với 14 lần vô địch.",
          "Juventus là đội lọt vào chung kết nhiều nhất với 20 lần.",
          "Đội vô địch sẽ được gắn một phù hiệu 3 màu (tiếng Ý: coccarda) giống như phù hiệu của lực lượng không quân, và được vào thẳng vòng bảng UEFA Europa League mùa giải tiếp theo."
        ]
      ],
      [
        "Giải thưởng Điện ảnh Viện Hàn lâm Anh Quốc",
        [
          "Giải thưởng Điện ảnh Viện Hàn lâm Anh quốc (còn gọi là Giải BAFTA) là giải thưởng được trao thường niên do Viện Hàn lâm Nghệ thuật Điện ảnh và Truyền hình Anh quốc (BAFTA).",
          "Giải thưởng này thường được coi là giải thưởng đồng cấp với giải Oscar ở Anh.Năm 2014, lễ trao giải được tổ chức tại trung tâm Luân Đôn ở Nhà hát opera Hoàng gia.",
          "Giải BAFTA lần thứ 68 được tổ chức vào ngày 8 tháng 2 năm 2015."
        ]
      ]
    ],
    "type": "bridge",
    "supporting_facts": [
      [
        "Giải Oscar",
        0
      ],
      [
        "Viện Hàn lâm Khoa học và Nghệ thuật Điện ảnh",
        0
      ]
    ]
  },
```

Thế mà trên github hotpotQA ghi nhầm là 

```bash
{
    "answer": "This is the answer",
    "context": {
        "sentences": [["Sent 1"], ["Sent 2"]],
        "title": ["Title1", "Title 2"]
    },
    "id": "000001",
    "level": "hard",
    "question": "What is the answer?",
    "supporting_facts": {
        "sent_id": [0, 1, 3],
        "title": ["Title of para 1", "Title of para 2", "Title of para 3"]
    },
    "type": "bridge"
}

```

### Field Descriptions

- `_id`: Unique identifier for the question
- `question`: The actual question text
- `supporting_facts`: List of [title, sentence_index] pairs that contain the information needed to answer the question
- `context`: List of [title, sentences] pairs where:
  - `title`: The title of the context paragraph
  - `sentences`: List of sentences in that paragraph
- `answer`: The correct answer to the question

### Example

```json
{
  "_id": "5a7a06935542990198eaf050",
  "question": "Which magazine was started first Arthur's Magazine or First for Women?",
  "supporting_facts": [
    ["Arthur's Magazine", 0],
    ["First for Women", 0]
  ],
  "context": [
    ["Arthur's Magazine", [
      "Arthur's Magazine was an American literary periodical published in the 1840s.",
      "It was founded by Timothy Shay Arthur in Philadelphia."
    ]],
    ["First for Women", [
      "First for Women is a women's magazine published by Bauer Media Group.",
      "It was launched in 1989."
    ]]
  ],
  "answer": "Arthur's Magazine"
}
```


----
# 2. Tại sao cần chuyển đổi kiểu data? 


## 🛠️ 2. **Về mặt kỹ thuật**

### Format gốc:

```json
"context": [
  ["Title1", ["Sent1", "Sent2"]],
  ["Title2", ["Sent1", "Sent2"]]
]
```

* ➕ Chi tiết, hỗ trợ kiểm tra từng câu
* ➖ Không phù hợp với các tokenizer/mô hình xử lý đoạn văn

### Format sau khi convert:

```json
"contexts": [
  {"title": "Title1", "paragraph_text": "Sent1 Sent2", "is_supporting": true},
  ...
]
```

* ➕ Dễ dùng với retriever, dễ `tokenize`, dễ batch
* ➖ Mất granular control nếu bạn muốn highlight từng câu

---


> **Không chuyển → vẫn dùng được nếu bạn làm reasoning trên từng câu (multi-hop QA truyền thống).**
> **Nhưng muốn kết hợp retrieval / huấn luyện reader / inference nhanh → phải chuyển.**


## Output Format (JSONL)

The output is in JSONL format (JSON Lines), where each line is a valid JSON object representing a converted question-answer pair.

### Structure

```json
{
  "question_id": "unique_question_id",
  "question": "The question text",
  "contexts": [
    {
      "idx": 0,
      "title": "title1",
      "paragraph_text": "Combined sentences from the paragraph",
      "is_supporting": true/false
    },
    ...
  ],
  "answer": "The answer text"
}
```

### Field Descriptions

- `question_id`: Same as input `_id`
- `question`: Same as input `question`
- `contexts`: List of context objects where each object contains:
  - `idx`: Index of the context in the list
  - `title`: Title of the context paragraph
  - `paragraph_text`: All sentences from the paragraph joined with spaces
  - `is_supporting`: Boolean indicating if this paragraph contains supporting facts
- `answer`: Same as input `answer`

### Example

```json
{
  "question_id": "5a7a06935542990198eaf050",
  "question": "Which magazine was started first Arthur's Magazine or First for Women?",
  "contexts": [
    {
      "idx": 0,
      "title": "Arthur's Magazine",
      "paragraph_text": "Arthur's Magazine was an American literary periodical published in the 1840s. It was founded by Timothy Shay Arthur in Philadelphia.",
      "is_supporting": true
    },
    {
      "idx": 1,
      "title": "First for Women",
      "paragraph_text": "First for Women is a women's magazine published by Bauer Media Group. It was launched in 1989.",
      "is_supporting": true
    }
  ],
  "answer": "Arthur's Magazine"
}
```

## Key Changes in Conversion

1. **Format Change**: From JSON array to JSONL (one JSON object per line)
2. **Context Restructuring**: 
   - Sentences are joined into a single paragraph text
   - Added `idx` field for ordering
   - Added `is_supporting` flag based on supporting_facts
3. **Field Renaming**: `_id` becomes `question_id`
4. **Simplification**: Supporting facts are converted to boolean flags in contexts


---
