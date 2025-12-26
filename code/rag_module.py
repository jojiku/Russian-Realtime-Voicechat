import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Generator
import time

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("⚠️ FAISS not installed. Run: pip install faiss-cpu")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("⚠️ sentence-transformers not installed. Run: pip install sentence-transformers")

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"


class RAGRetriever:
    """Fast semantic search using FAISS and local embeddings"""
    
    def __init__(
        self,
        documents_path: str = "resources/rag/documents.json",
        embeddings_path: str = "resources/rag/embeddings.npy",
        model_name: str = EMBEDDING_MODEL,
        top_k: int = 2,
        similarity_threshold: float = 0.7,
        min_similarity_for_injection: float = 0.8
    ):
        self.documents_path = Path(documents_path)
        self.embeddings_path = Path(embeddings_path)
        self.model_name = model_name
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.min_similarity_for_injection = min_similarity_for_injection

        self.documents: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.Index] = None
        self.model: Optional[SentenceTransformer] = None
        self.is_initialized: bool = False
    
    def initialize(self) -> bool:
        """Load model, documents, embeddings, and build FAISS index"""
        if not FAISS_AVAILABLE:
            logger.error("❌ FAISS not available")
            return False
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("❌ sentence-transformers not available")
            return False
        
        if not self.documents_path.exists() or not self.embeddings_path.exists():
            logger.error(f"❌ Missing files. Run rag_setup.py first!")
            return False
        
        try:
            # Load model
            logger.info(f"📥 Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"✅ Model loaded")
            
            # Load documents
            logger.info(f"📖 Loading documents from {self.documents_path}")
            with open(self.documents_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            logger.info(f"✅ Loaded {len(self.documents)} documents")
            
            # Load embeddings
            logger.info(f"📊 Loading embeddings from {self.embeddings_path}")
            self.embeddings = np.load(self.embeddings_path).astype(np.float32)
            logger.info(f"✅ Loaded embeddings: {self.embeddings.shape}")
            
            # Normalize for cosine similarity
            faiss.normalize_L2(self.embeddings)
            
            # Build FAISS index
            embedding_dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(embedding_dim)
            self.index.add(self.embeddings)
            logger.info(f"✅ Built FAISS index with {self.index.ntotal} vectors")
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize RAG: {e}", exc_info=True)
            return False
    
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a query"""
        # For E5 models, add "query: " prefix
        if 'e5' in self.model_name.lower():
            query = f"query: {query}"
        
        embedding = self.model.encode(query, convert_to_numpy=True).astype(np.float32)
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.reshape(1, -1)
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """Search for relevant documents"""
        if not self.is_initialized:
            logger.warning("⚠️ RAG not initialized")
            return []
        
        k = top_k or self.top_k
        
        query_embedding = self._get_query_embedding(query)
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            if score < self.similarity_threshold:
                continue
            
            doc = self.documents[idx].copy()
            doc['similarity_score'] = float(score)
            results.append(doc)
        
        # Log results
        logger.info(f"🔍 RAG Search: '{query[:50]}...'")
        if results:
            logger.info(f"🔍 Found {len(results)} relevant docs (threshold={self.similarity_threshold}):")
            for i, doc in enumerate(results[:3], 1):
                doc_type = doc.get('metadata', {}).get('type', 'unknown')
                logger.info(f"   {i}.[{doc['similarity_score']:.3f}][{doc_type}] {doc['content'][:60]}...")
        else:
            logger.info(f"🔍 No documents above threshold {self.similarity_threshold}")
        
        return results
    
    def should_inject_context(self, results: List[Dict]) -> bool:
        """
        Determine if RAG context should be injected based on result quality.
        Returns True only if we have highly relevant results.
        """
        if not results:
            return False
        
        # Check if the best result is above the minimum threshold
        best_score = max(r['similarity_score'] for r in results)
        
        if best_score >= self.min_similarity_for_injection:
            logger.info(f"🔍 RAG: Best match score {best_score:.3f} >= {self.min_similarity_for_injection} ✅ Injecting context")
            return True
        else:
            logger.info(f"🔍 RAG: Best match score {best_score:.3f} < {self.min_similarity_for_injection} ❌ Skipping injection")
            return False

    def get_context_string(self, query: str, max_chars: int = 1500) -> str:
        """Get formatted context string for LLM prompt injection"""
        rag_start_time = time.time()
        results = self.search(query)
        
        if not self.should_inject_context(results):
            logger.info("🔍 RAG: Query not related to car database, using LLM knowledge only")
            return ""
        
        # Group results by type for better organization
        by_type = {}
        for doc in results:
            doc_type = doc.get('metadata', {}).get('type', 'unknown')
            if doc_type not in by_type:
                by_type[doc_type] = []
            by_type[doc_type].append(doc)
        
        context_parts = []
        total_len = 0
        
        # Priority order for different types
        type_priority = [
            'car_unit',           # Actual cars in stock
            'model',              # Model info
            'brand',              # Brand info
            'dealership_hours',   # Hours
            'dealership_location',# Location
            'dealership_contact', # Contacts
            'dealership_services',# Services
            'dealership_offers',  # Special offers
            'trade_in',           # Trade-in info
            'test_drive',         # Test drive info
            'recommendation',     # Recommendations
            'comparison',         # Comparisons
            'schedule'            # Viewing schedule
        ]
        
        # Add documents in priority order
        for doc_type in type_priority:
            if doc_type not in by_type:
                continue
            
            for doc in by_type[doc_type]:
                content = doc['content']
                if total_len + len(content) > max_chars:
                    break
                
                # Format based on type for better readability
                if doc_type == 'car_unit':
                    context_parts.append(f"🚗 АВТОМОБИЛЬ: {content}")
                elif doc_type in ['dealership_hours', 'dealership_location', 'dealership_contact']:
                    context_parts.append(f"ℹ️ {content}")
                elif doc_type == 'schedule':
                    context_parts.append(f"📅 {content}")
                elif doc_type in ['recommendation', 'comparison']:
                    context_parts.append(f"💡 {content}")
                else:
                    context_parts.append(f"• {content}")
                
                total_len += len(content)
            
            if total_len >= max_chars:
                break
        
        if not context_parts:
            return ""
        
        context = "\n".join(context_parts)
        
        # Updated format - tells LLM to skip greeting and use the data
        formatted = f"""
=== ИНФОРМАЦИЯ ИЗ БАЗЫ ДАННЫХ ===
{context}
=== КОНЕЦ ИНФОРМАЦИИ ===

ИНСТРУКЦИЯ: Отвечай кратко и по существу. НЕ начинай с приветствия - клиент уже поприветствован. Используй информацию выше для ответа.
"""
        logger.info(f"🔍 RAG: Injecting {len(context_parts)} documents ({total_len} chars)")
        rag_end_time = time.time()
        logger.info(f"🔍 RAG TIME TOOK: {rag_end_time - rag_start_time:.3f}s")
        return formatted


class RAGEnhancedLLM:
    """Wrapper that enhances LLM calls with RAG context"""
    
    def __init__(self, llm, rag_retriever: RAGRetriever, inject_context: bool = True):
        self.llm = llm
        self.rag = rag_retriever
        self.inject_context = inject_context
        self.backend = getattr(llm, 'backend', 'unknown')
        self.model = getattr(llm, 'model', 'unknown')
    
    def generate(
        self,
        text: str,
        history: Optional[List[Dict[str, str]]] = None,
        use_system_prompt: bool = True,
        request_id: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Generate with automatic RAG context injection"""
        enhanced_text = text
        
        if self.inject_context and self.rag.is_initialized:
            context = self.rag.get_context_string(text)
            
            if context:
                # RAG found relevant info - inject it
                enhanced_text = f"""{context}

Вопрос клиента: {text}

Отвечай сразу по существу, БЕЗ приветствия (клиент уже поприветствован)."""
                logger.info(f"🔍 RAG: Context injected into prompt")
            else:
                logger.info(f"🔍 RAG: No context found, using original prompt")
        
        yield from self.llm.generate(
            text=enhanced_text,
            history=history,
            use_system_prompt=use_system_prompt,
            request_id=request_id,
            **kwargs
        )
    
    def __getattr__(self, name):
        return getattr(self.llm, name)
    
    def prewarm(self, *args, **kwargs):
        return self.llm.prewarm(*args, **kwargs)
    
    def measure_inference_time(self, *args, **kwargs):
        return self.llm.measure_inference_time(*args, **kwargs)
    
    def cancel_generation(self, *args, **kwargs):
        return self.llm.cancel_generation(*args, **kwargs)


def create_rag_retriever(
    documents_path: str = "resources/rag/documents.json",
    embeddings_path: str = "resources/rag/embeddings.npy",
) -> Optional[RAGRetriever]:
    """Create and initialize a RAG retriever"""
    if not FAISS_AVAILABLE or not SENTENCE_TRANSFORMERS_AVAILABLE:
        logger.error("❌ Missing dependencies")
        return None
    
    retriever = RAGRetriever(
        documents_path=documents_path,
        embeddings_path=embeddings_path
    )
    
    if retriever.initialize():
        logger.info("✅ RAG Retriever initialized successfully")
        return retriever
    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("=" * 60)
    print("RAG Module Test - Smart Filtering")
    print("=" * 60)
    
    retriever = create_rag_retriever()
    
    if not retriever:
        print("❌ Failed")
        exit(1)
    
    # Test queries - comprehensive coverage of all document types
    test_queries = [
        # Car units (should trigger RAG)
        ("Есть ли у вас Toyota Camry?", True),
        ("Покажите седаны до 3 миллионов", True),
        ("Какие машины с пробегом меньше 50000 км?", True),
        
        # Models and brands (should trigger RAG)
        ("Какие японские машины есть?", True),
        ("Расскажите про Honda Civic", True),
        ("Что за бренд Hyundai?", True),
        
        # Dealership info (should trigger RAG)
        ("До скольки вы работаете?", True),
        ("Как к вам доехать?", True),
        ("Какой у вас адрес?", True),
        ("Какие услуги вы предлагаете?", True),
        ("Есть ли у вас trade-in?", True),
        
        # Schedule (should trigger RAG)
        ("Когда можно приехать на осмотр?", True),
        ("Какие слоты свободны завтра?", True),
        
        # Recommendations (should trigger RAG)
        ("Что посоветуете для семьи?", True),
        ("Чем седан отличается от внедорожника?", True),
        ("Какую машину взять для города?", True),
        
        # Non-car queries (should NOT trigger RAG)
        ("Какая сегодня погода в Москве?", False),
        ("Как приготовить борщ?", False),
        ("Кто президент России?", False),
    ]
    
    print("\n" + "=" * 60)
    print("Testing Smart RAG Filtering")
    print("=" * 60)
    
    correct = 0
    total = len(test_queries)
    
    for query, expected_rag in test_queries:
        print(f"\n🔍 Query: {query}")
        print(f"   Expected RAG: {'Yes' if expected_rag else 'No'}")
        
        context = retriever.get_context_string(query)
        actual_rag = bool(context)
        
        status = "✅" if actual_rag == expected_rag else "❌"
        if actual_rag == expected_rag:
            correct += 1
        
        print(f"   Actual RAG: {'Yes' if actual_rag else 'No'} {status}")
        
        if context:
            preview = context.replace("\n", " ")[:150]
            print(f"   Context preview: {preview}...")
    
    print("\n" + "=" * 60)
    print(f"Results: {correct}/{total} correct ({100*correct//total}%)")
    print("=" * 60)