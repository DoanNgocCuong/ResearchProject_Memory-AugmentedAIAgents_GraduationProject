"""
Test retrieval functionality using FAISS vector store.
"""

from pathlib import Path
import sys
import logging
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

# Setup paths
current_file = Path(__file__)
retrieval_dir = current_file.parent
sys.path.append(str(retrieval_dir))

# Import retrievers and other modules
from retrievers.vector_retriever import VectorRetriever
from retrievers.bm25_retriever import BM25Retriever
from retrievers.hybrid_retriever import HybridRetriever
from config import HUGGINGFACE_API_KEY

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize embedding model using API
logger.info("Initializing embedding model via API...")
EMBEDDING_MODEL = HuggingFaceInferenceAPIEmbeddings(
    api_key=HUGGINGFACE_API_KEY,
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

def get_test_documents() -> List[Document]:
    """Create and return test documents."""
    logger.info("Creating test documents...")
    documents = [
        Document(
            page_content="The quick brown fox jumps over the lazy dog",
            metadata={"source": "test1.txt"}
        ),
        Document(
            page_content="A fast orange fox leaps across a sleepy canine",
            metadata={"source": "test2.txt"}
        ),
        Document(
            page_content="The weather is beautiful today",
            metadata={"source": "test3.txt"}
        )
    ]
    logger.info(f"Created {len(documents)} test documents")
    return documents

def test_vector_retriever(documents: List[Document]) -> None:
    """Test vector retriever functionality with FAISS."""
    logger.info("\n=== Testing Vector Retriever with FAISS ===")
    try:
        logger.info("Creating FAISS vector store...")
        vector_store = FAISS.from_documents(documents, EMBEDDING_MODEL)
        logger.info("Initializing VectorRetriever...")
        vector_retriever = VectorRetriever(vector_store)
        logger.info("Testing vector search with query: 'fox jumping'")
        vector_results = vector_retriever.retrieve_documents("fox jumping")
        logger.info(f"Vector search found {len(vector_results)} results")
        print(f"Vector results: {[doc.page_content for doc in vector_results]}")
    except Exception as e:
        logger.error(f"Vector retriever test failed: {str(e)}", exc_info=True)

def test_bm25_retriever(documents: List[Document]) -> None:
    """Test BM25 retriever functionality."""
    logger.info("\n=== Testing BM25 Retriever ===")
    try:
        logger.info("Initializing BM25Retriever...")
        bm25_retriever = BM25Retriever(documents)
        logger.info("Testing BM25 search with query: 'fox jumping'")
        bm25_results = bm25_retriever.retrieve_documents("fox jumping")
        logger.info(f"BM25 search found {len(bm25_results)} results")
        print(f"BM25 results: {[doc.page_content for doc in bm25_results]}")
    except Exception as e:
        logger.error(f"BM25 retriever test failed: {str(e)}", exc_info=True)

def test_hybrid_retriever(documents: List[Document], vector_store: FAISS) -> None:
    """Test hybrid retriever functionality."""
    logger.info("\n=== Testing Hybrid Retriever with FAISS ===")
    try:
        logger.info("Initializing HybridRetriever...")
        hybrid_retriever = HybridRetriever(
            vector_store=vector_store,
            documents=documents
        )
        logger.info("Testing hybrid search with query: 'fox jumping'")
        hybrid_results = hybrid_retriever.retrieve_documents("fox jumping")
        logger.info(f"Hybrid search found {len(hybrid_results)} results")
        print(f"Hybrid results: {[doc.page_content for doc in hybrid_results]}")
    except Exception as e:
        logger.error(f"Hybrid retriever test failed: {str(e)}", exc_info=True)

def test_retrieval() -> None:
    """Test different retrieval methods with FAISS."""
    logger.info("Starting FAISS retrieval tests...")
    
    # Get test documents
    documents = get_test_documents()
    
    # Initialize vector store for hybrid retriever
    vector_store = FAISS.from_documents(documents, EMBEDDING_MODEL)
    
    # Run tests
    test_vector_retriever(documents)
    test_bm25_retriever(documents)
    test_hybrid_retriever(documents, vector_store)
    
    logger.info("\n=== FAISS Retrieval Tests Completed ===")

if __name__ == "__main__":
    try:
        logger.info("Starting FAISS test execution...")
        test_retrieval()
        logger.info("FAISS test execution completed successfully")
    except Exception as e:
        logger.error(f"FAISS test execution failed: {str(e)}", exc_info=True) 