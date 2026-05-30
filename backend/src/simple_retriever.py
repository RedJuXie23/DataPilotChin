"""
Simple retriever for styling instructions - replaces the full ChromaDB-based retriever.
"""
from typing import List


class SimpleRetriever:
    """A simple text retriever that does keyword matching instead of vector search."""
    
    def __init__(self, documents: List[str]):
        self.documents = documents
    
    def retrieve(self, query: str, k: int = 1) -> List[str]:
        """Return top-k documents matching query keywords."""
        query_lower = query.lower()
        scored = []
        for doc in self.documents:
            # Simple keyword overlap score
            doc_lower = doc.lower()
            score = sum(1 for word in query_lower.split() if word in doc_lower)
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [doc for _, doc in scored[:k]]
        
        # If no match, return first document as fallback
        if not results and self.documents:
            results = [self.documents[0]]
        
        return results
    
    def as_retriever(self, similarity_top_k: int = 1):
        """Compatibility method - returns self."""
        self._k = similarity_top_k
        return self
