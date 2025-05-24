Refactor file `retriever.py` để code sạch và dễ bảo trì hơn. Chúng ta sẽ chia thành các module nhỏ hơn và sử dụng các design pattern phù hợp:

1. Đầu tiên, tạo cấu trúc thư mục mới:
```
layers/
  _04_retrieval/
    __init__.py
    retriever.py
    vector_stores/
      __init__.py
      base.py
      faiss_store.py
      qdrant_store.py
    retrievers/
      __init__.py
      base.py
      vector_retriever.py
      bm25_retriever.py
      hybrid_retriever.py
      compression_retriever.py
    config.py
```

2. Tạo file `config.py` để quản lý cấu hình:
```python
"""
Configuration settings for the retrieval system.
"""

from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Default configuration
DEFAULT_K = 4
DEFAULT_HYBRID_WEIGHTS = [0.7, 0.3]  # [vector_weight, keyword_weight]
DEFAULT_VECTOR_STORE_TYPE = "faiss"

# Embeddings configuration
DEFAULT_EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Qdrant configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
DEFAULT_COLLECTION_NAME = "documents"
```

3. Tạo file `vector_stores/base.py`:
```python
"""
Base class for vector stores.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def create_store(self, documents: List[Document]) -> VectorStore:
        """Create a vector store from documents."""
        pass
    
    @abstractmethod
    def get_retriever(self, k: int) -> VectorStore:
        """Get a retriever from the vector store."""
        pass
```

4. Tạo file `vector_stores/qdrant_store.py`:
```python
"""
Qdrant vector store implementation.
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from .base import BaseVectorStore
from ..config import QDRANT_URL, QDRANT_API_KEY, DEFAULT_COLLECTION_NAME

class QdrantStore(BaseVectorStore):
    """Qdrant vector store implementation."""
    
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_function = None
    ):
        self.url = url or QDRANT_URL
        self.api_key = api_key or QDRANT_API_KEY
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.client = None
        self.store = None
        
        if not self.url or not self.api_key:
            raise ValueError("Qdrant URL and API key are required")
            
        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key
        )
    
    def create_store(self, documents: List[Document]) -> VectorStore:
        """Create a Qdrant vector store from documents."""
        self.store = Qdrant.from_documents(
            documents=documents,
            embedding=self.embedding_function,
            client=self.client,
            collection_name=self.collection_name
        )
        return self.store
    
    def get_retriever(self, k: int) -> VectorStore:
        """Get a retriever from the Qdrant store."""
        if not self.store:
            self.store = Qdrant(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embedding_function
            )
        return self.store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
```

5. Tạo file `retrievers/base.py`:
```python
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
```

6. Tạo file `retrievers/vector_retriever.py`:
```python
"""
Vector-based retriever implementation.
"""

from typing import List, Optional
from langchain_core.documents import Document
from .base import BaseRetriever
from ..vector_stores.base import BaseVectorStore

class VectorRetriever(BaseRetriever):
    """Vector-based retriever implementation."""
    
    def __init__(
        self,
        vector_store: BaseVectorStore,
        k: int = 4
    ):
        self.vector_store = vector_store
        self.k = k
        self.retriever = vector_store.get_retriever(k)
    
    def retrieve_documents(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[Document]:
        """Retrieve documents using vector search."""
        if k is not None:
            self.retriever.search_kwargs["k"] = k
        return self.retriever.get_relevant_documents(query)
```

7. Cuối cùng, cập nhật file `retriever.py` chính:
```python
"""
Main retriever module that combines different retrieval strategies.
"""

from typing import List, Dict, Any, Optional, Union
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from .config import (
    DEFAULT_K,
    DEFAULT_HYBRID_WEIGHTS,
    DEFAULT_VECTOR_STORE_TYPE,
    DEFAULT_EMBEDDINGS
)
from .vector_stores.qdrant_store import QdrantStore
from .retrievers.vector_retriever import VectorRetriever
from .retrievers.bm25_retriever import BM25Retriever
from .retrievers.hybrid_retriever import HybridRetriever
from .retrievers.compression_retriever import CompressionRetriever

class DocumentRetriever:
    """
    Main retriever class that combines different retrieval strategies.
    """
    
    def __init__(
        self,
        retriever_type: str = "vector",
        vector_store_type: str = DEFAULT_VECTOR_STORE_TYPE,
        documents: Optional[List[Document]] = None,
        embeddings_model = None,
        hybrid_weights: List[float] = DEFAULT_HYBRID_WEIGHTS,
        k: int = DEFAULT_K,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        collection_name: str = "documents"
    ):
        """
        Initialize the document retriever.
        
        Args:
            retriever_type: Type of retriever to use
            vector_store_type: Type of vector store to use
            documents: List of documents for BM25 search
            embeddings_model: Embeddings model to use
            hybrid_weights: Weights for hybrid search
            k: Number of documents to return
            qdrant_url: URL for Qdrant server
            qdrant_api_key: API key for Qdrant
            collection_name: Name of the Qdrant collection
        """
        self.retriever_type = retriever_type
        self.vector_store_type = vector_store_type
        self.documents = documents
        self.embeddings_model = embeddings_model or DEFAULT_EMBEDDINGS
        self.hybrid_weights = hybrid_weights
        self.k = k
        
        # Initialize vector store
        if vector_store_type == "qdrant":
            self.vector_store = QdrantStore(
                url=qdrant_url,
                api_key=qdrant_api_key,
                collection_name=collection_name,
                embedding_function=self.embeddings_model.embed_query
            )
        
        # Initialize retriever
        self._initialize_retriever()
    
    def _initialize_retriever(self):
        """Initialize the appropriate retriever based on type."""
        if self.retriever_type == "vector":
            self.retriever = VectorRetriever(
                vector_store=self.vector_store,
                k=self.k
            )
        elif self.retriever_type == "bm25":
            self.retriever = BM25Retriever(
                documents=self.documents,
                k=self.k
            )
        elif self.retriever_type == "hybrid":
            self.retriever = HybridRetriever(
                vector_store=self.vector_store,
                documents=self.documents,
                weights=self.hybrid_weights,
                k=self.k
            )
        elif self.retriever_type == "compression":
            self.retriever = CompressionRetriever(
                vector_store=self.vector_store,
                k=self.k
            )
        else:
            raise ValueError(f"Unsupported retriever type: {self.retriever_type}")
    
    def retrieve_documents(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[Document]:
        """Retrieve documents relevant to the query."""
        return self.retriever.retrieve_documents(query, k)
```

Lợi ích của cách refactor này:

1. **Tách biệt các thành phần:**
   - Mỗi loại vector store có class riêng
   - Mỗi loại retriever có class riêng
   - Cấu hình được tách ra file riêng

2. **Dễ mở rộng:**
   - Thêm vector store mới chỉ cần tạo class mới kế thừa `BaseVectorStore`
   - Thêm retriever mới chỉ cần tạo class mới kế thừa `BaseRetriever`

3. **Dễ test:**
   - Có thể test từng component riêng biệt
   - Dễ dàng mock các dependencies

4. **Dễ bảo trì:**
   - Code được tổ chức rõ ràng
   - Mỗi class có một nhiệm vụ cụ thể
   - Dễ dàng thay đổi implementation

5. **Linh hoạt:**
   - Có thể dễ dàng chuyển đổi giữa các loại vector store
   - Có thể kết hợp nhiều loại retriever khác nhau


---


# You are CTO, bình tĩnh để kiểm tra toàn bộ hệ thống để xem vấn đề thực sự nằm ở đâu. Kiểm tra thật kỹ mọi thứ để lên plan chi tiết rồi mới coding @_04_retrieval 

```bash
_04_retrieval/
├── __init__.py          # Exports các class chính
├── config.py            # Cấu hình mặc định
├── retriever.py         # Class DocumentRetriever chính
├── test_retriever.py    # Test cases
├── retrievers/          # Các retriever cụ thể
│   ├── __init__.py
│   ├── base.py          # Base class cho retrievers
│   ├── vector_retriever.py
│   ├── bm25_retriever.py
│   ├── hybrid_retriever.py
│   └── compression_retriever.py
└── vector_stores/       # Các vector store
    ├── __init__.py
    ├── base.py          # Base class cho vector stores
    └── qdrant_store.py  # Qdrant implementation
```