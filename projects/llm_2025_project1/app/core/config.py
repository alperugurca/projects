from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "CV Analyzer"
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL_NAME: str = "gpt-4"  # Using GPT-4 for better analysis
    
    # Vector Store Configuration
    CHROMA_PERSIST_DIRECTORY: str = "data/chroma"
    
    # Retrieval Configuration
    HYBRID_ALPHA: float = 0.5  # Weight for hybrid search (0 = BM25 only, 1 = Vector only)
    RERANK_TOP_K: int = 5
    
    # Analysis Configuration
    MAX_TOKENS: int = 2000
    TEMPERATURE: float = 0.7
    
    class Config:
        case_sensitive = True