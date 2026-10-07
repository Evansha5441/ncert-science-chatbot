import os
import json
import re
from typing import List, Tuple
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import settings

class RAGRetriever:
    _instance = None

    def __init__(self):
        print(f"Loading FAISS vector store from {settings.INDEX_PATH}...")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = FAISS.load_local(
            settings.INDEX_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print("Vector store loaded successfully!")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = RAGRetriever()
        return cls._instance

    def retrieve(self, query: str, top_k: int = None) -> Tuple[str, List[str], List[float]]:
        """
        Retrieves relevant context chunks and chapter citations from NCERT Class 10 Science.
        Returns:
            context_str: str of concatenated chunks
            citations: list of unique chapter names
            scores: similarity scores (L2 distance)
        """
        k = top_k or settings.TOP_K_CHUNKS
        # faiss.similarity_search_with_score returns (doc, distance)
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        context_chunks = []
        citations = []
        scores = []

        for doc, score in results:
            ch_name = doc.metadata.get("chapter_name", "NCERT Class 10 Science")
            if ch_name not in citations:
                citations.append(ch_name)
            context_chunks.append(f"[{ch_name}]: {doc.page_content}")
            scores.append(float(score))

        context_str = "\n\n".join(context_chunks)
        return context_str, citations, scores
