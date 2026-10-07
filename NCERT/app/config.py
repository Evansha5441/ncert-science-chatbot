import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    LLM_PROVIDER: str = "groq"
    GROQ_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    OPENAI_API_KEY: str = ""

    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GEMINI_MODEL: str = "gemini-1.5-flash"
    OPENAI_MODEL: str = "gpt-4o-mini"

    INDEX_PATH: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "faiss_index")
    CACHE_DB_PATH: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache.db")
    CACHE_INDEX_PATH: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache_faiss")

    # High precision cosine similarity threshold to avoid false-positive semantic matches
    CACHE_SIMILARITY_THRESHOLD: float = 0.91
    
    # Retrieval configuration
    TOP_K_CHUNKS: int = 4

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
