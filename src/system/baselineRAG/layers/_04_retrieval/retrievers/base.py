"""
Base class for retrievers.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from langchain_core.documents import Document

class BaseRetriever(ABC):
    """Abstract base class for retrievers."""
    
    @abstractmethod
    def retrieve_documents(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[Document]:
        """Retrieve documents relevant to the query."""
        pass 