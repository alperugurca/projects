from typing import List, Dict
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi
from ..core.config import Settings
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class RetrievalService:
    """
    Handles hybrid retrieval combining vector search and BM25
    """
    
    def __init__(self):
        self.settings = Settings()
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.settings.OPENAI_API_KEY)
        self.vectorstore = self._initialize_vectorstore()
        self.bm25 = None
        self.corpus = self._load_corpus()
        
    def _initialize_vectorstore(self) -> Chroma:
        """
        Initialize and return ChromaDB vector store
        """
        return Chroma(
            persist_directory=self.settings.CHROMA_PERSIST_DIRECTORY,
            embedding_function=self.embeddings
        )
        
    def _load_corpus(self) -> List[str]:
        """
        Load the text corpus for BM25
        """
        # In a real application, this would load from your knowledge base
        # For now, we'll return a sample corpus
        return [
            "Strong technical skills and programming experience",
            "Leadership and project management expertise",
            "Communication and interpersonal abilities",
            "Problem-solving and analytical thinking",
            # Add more relevant documents
        ]
        
    def _initialize_bm25(self):
        """
        Initialize BM25 with tokenized corpus
        """
        tokenized_corpus = [doc.split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """
        Perform hybrid retrieval combining vector search and BM25
        """
        # Get vector search results
        vector_results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        # Get BM25 results
        if self.bm25 is None:
            self._initialize_bm25()
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize scores
        scaler = MinMaxScaler()
        vector_scores = np.array([score for _, score in vector_results]).reshape(-1, 1)
        normalized_vector_scores = scaler.fit_transform(vector_scores).flatten()
        normalized_bm25_scores = scaler.fit_transform(bm25_scores.reshape(-1, 1)).flatten()
        
        # Combine scores using weighted average
        alpha = self.settings.HYBRID_ALPHA
        combined_scores = (alpha * normalized_vector_scores + 
                         (1 - alpha) * normalized_bm25_scores)
        
        # Sort and return top results
        top_indices = np.argsort(combined_scores)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'content': self.corpus[idx],
                'score': combined_scores[idx]
            })
            
        return self._rerank_results(query, results)
        
    def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        Rerank results using a more sophisticated model
        """
        # In a real application, you would use a reranking model like Cohere Rerank
        # For now, we'll just return the results as is
        return results[:self.settings.RERANK_TOP_K]