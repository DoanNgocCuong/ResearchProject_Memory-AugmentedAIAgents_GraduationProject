import json

def convert_to_vietnamese_readable(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    convert_to_vietnamese_readable('train.json', 'train_vi.json')
    print("Đã chuyển mã hóa xong! File mới: train_vi.json")
    convert_to_vietnamese_readable('dev.json', 'dev_vi.json')
    print("Đã chuyển mã hóa xong! File mới: dev_vi.json")
    convert_to_vietnamese_readable('test.json', 'test_vi.json')
    print("Đã chuyển mã hóa xong! File mới: test_vi.json")
