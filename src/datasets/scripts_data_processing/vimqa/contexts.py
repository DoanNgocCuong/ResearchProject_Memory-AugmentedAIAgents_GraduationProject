from typing import Dict, List, Any, Set, Tuple

def create_contexts_gold(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create contexts_gold.jsonl from VIMQA data.
    Contains question, answer, and labeled contexts.
    
    Args:
        data: List of VIMQA examples
        
    Returns:
        List of contexts with gold labels
    """
    contexts_gold = []
    
    for example in data:
        # Create set of supporting facts for quick lookup
        supporting_facts = {(title, sent_id) for title, sent_id in example["supporting_facts"]}
        
        # Process contexts
        contexts = []
        for title, sentences in example["context"]:
            for sent_idx, sentence in enumerate(sentences):
                is_supporting = (title, sent_idx) in supporting_facts
                contexts.append({
                    "title": title,
                    "text": sentence,
                    "is_supporting": is_supporting
                })
        
        contexts_gold.append({
            "question_id": example["_id"],
            "question": example["question"],
            "answer": example["answer"],
            "contexts": contexts
        })
    
    return contexts_gold 