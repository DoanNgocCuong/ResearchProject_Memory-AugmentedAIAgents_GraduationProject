from typing import Dict, List, Any

def create_qa_pairs(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create qa_pairs.jsonl from VIMQA data.
    Contains question, answer, and supporting facts.
    
    Args:
        data: List of VIMQA examples
        
    Returns:
        List of QA pairs
    """
    qa_pairs = []
    
    for example in data:
        qa_pair = {
            "question_id": example["_id"],
            "question": example["question"],
            "answer": example["answer"],
            "supporting_facts": example["supporting_facts"]
        }
        qa_pairs.append(qa_pair)
    
    return qa_pairs 