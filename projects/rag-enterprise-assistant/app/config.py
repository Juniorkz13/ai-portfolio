import os
from functools import lru_cache


@lru_cache
def get_settings():
    return {
        "env": os.getenv("ENV", "dev"),
        "gemini_api_key": os.getenv("GEMINI_API_KEY"),
        "gemini_model": os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
        "rag_top_k": int(os.getenv("RAG_TOP_K", 3)),
        "cache_size": int(os.getenv("CACHE_SIZE", 128)),
    }