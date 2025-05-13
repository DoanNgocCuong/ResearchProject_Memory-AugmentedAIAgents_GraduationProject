import json
import os
from datasets.CKP.convert_hotpotqa import convert_hotpotqa_example, process_hotpotqa_file

# Sample input data
sample_input = [
    {
        "_id": "5a7a06935542990198eaf050",
        "question": "Which magazine was started first Arthur's Magazine or First for Women?",
        "supporting_facts": [
            ["Arthur's Magazine", 0],
            ["First for Women", 0]
        ],
        "context": [
            ["Arthur's Magazine", ["Arthur's Magazine was an American literary periodical published in the 1840s.", "It was founded by Timothy Shay Arthur in Philadelphia."]],
            ["First for Women", ["First for Women is a women's magazine published by Bauer Media Group.", "It was launched in 1989."]]
        ],
        "answer": "Arthur's Magazine"
    }
]

# Expected output format
expected_output = {
    "question_id": "5a7a06935542990198eaf050",
    "question": "Which magazine was started first Arthur's Magazine or First for Women?",
    "contexts": [
        {
            "idx": 0,
            "title": "Arthur's Magazine",
            "paragraph_text": "Arthur's Magazine was an American literary periodical published in the 1840s. It was founded by Timothy Shay Arthur in Philadelphia.",
            "is_supporting": True
        },
        {
            "idx": 1,
            "title": "First for Women",
            "paragraph_text": "First for Women is a women's magazine published by Bauer Media Group. It was launched in 1989.",
            "is_supporting": True
        }
    ],
    "answer": "Arthur's Magazine"
}

def test_convert_hotpotqa_example():
    """Test the conversion of a single example"""
    result = convert_hotpotqa_example(sample_input[0])
    assert result == expected_output, "Conversion of single example failed"
    print("✅ Single example conversion test passed!")

def test_process_hotpotqa_file():
    """Test the file processing functionality"""
    # Create temporary files
    input_file = "temp_input.json"
    output_file = "temp_output.jsonl"
    
    try:
        # Write sample input to file
        with open(input_file, 'w', encoding='utf-8') as f:
            json.dump(sample_input, f)
        
        # Process the file
        process_hotpotqa_file(input_file, output_file)
        
        # Read and verify output
        with open(output_file, 'r', encoding='utf-8') as f:
            result = json.loads(f.readline())
            assert result == expected_output, "File processing test failed"
        
        print("✅ File processing test passed!")
    
    finally:
        # Clean up temporary files
        if os.path.exists(input_file):
            os.remove(input_file)
        if os.path.exists(output_file):
            os.remove(output_file)

if __name__ == "__main__":
    print("Running tests...")
    test_convert_hotpotqa_example()
    test_process_hotpotqa_file()
    print("All tests completed!") 