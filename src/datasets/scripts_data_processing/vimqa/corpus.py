from typing import Dict, List, Any, Set

def create_corpus(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create corpus.jsonl from VIMQA data.
    Each document is a paragraph from the context.
    
    Args:
        data: List of VIMQA examples
        
    Returns:
        List of corpus documents
    """
    corpus = []
    seen_docs = set()  # To avoid duplicates
    
    for example in data:
        for title, sentences in example["context"]:
            doc_id = f"{title}_{len(corpus)}"  # Unique ID for each document
            if doc_id not in seen_docs:
                text = " ".join(sentences)
                corpus.append({
                    "doc_id": doc_id,
                    "title": title,
                    "text": text
                })
                seen_docs.add(doc_id)
    
    return corpus 