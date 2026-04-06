"""
Configuration module for Agentic RAG AI Research Scientist.
Loads environment variables and provides centralized config access.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file in the same directory as this script
_env_path = Path(__file__).parent / ".env"

# Try to load .env file
if _env_path.exists():
    _loaded = load_dotenv(dotenv_path=str(_env_path), override=True)
    print(f"[CONFIG] Loaded .env from: {_env_path}")
else:
    _loaded = False
    print(f"[CONFIG] WARNING: .env file not found at: {_env_path}")

# Fallback: If GROQ_API_KEY still not set, try reading .env directly
if not os.getenv("GROQ_API_KEY") and _env_path.exists():
    print("[CONFIG] Fallback: Reading .env file directly...")
    with open(_env_path, 'r', encoding='utf-8-sig') as f:  # utf-8-sig handles BOM
        content = f.read()
        print(f"[CONFIG] File has {len(content)} characters")
        for line in content.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and '=' in stripped:
                key, value = stripped.split('=', 1)
                key = key.strip()
                value = value.strip()
                print(f"[CONFIG] Found: {key}={value[:10]}..." if len(value) > 10 else f"[CONFIG] Found: {key}={value}")
                os.environ[key] = value
    print(f"[CONFIG] GROQ_API_KEY set: {'Yes' if os.getenv('GROQ_API_KEY') else 'No'}")


class Config:
    """Centralized configuration management."""
    
    @property
    def GROQ_API_KEY(self) -> str:
        return os.getenv("GROQ_API_KEY", "")
    
    @property
    def FAST_MODEL(self) -> str:
        return os.getenv("FAST_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
    
    @property
    def REASONING_MODEL(self) -> str:
        return os.getenv("REASONING_MODEL", "llama-3.3-70b-versatile")
    
    @property
    def DEV_MODE(self) -> bool:
        return os.getenv("DEV_MODE", "false").lower() == "true"
    
    @property
    def DEFAULT_PAPERS_K(self) -> int:
        return int(os.getenv("DEFAULT_PAPERS_K", "5"))
    
    @property
    def DEFAULT_CHUNKS_TOP_N(self) -> int:
        return int(os.getenv("DEFAULT_CHUNKS_TOP_N", "10"))
    
    @property
    def EMBEDDING_MODEL(self) -> str:
        return "all-MiniLM-L6-v2"
    
    @property
    def CHUNK_SIZE(self) -> int:
        return 500
    
    @property
    def CHUNK_OVERLAP(self) -> int:
        return 50
    
    @property
    def LOG_LEVEL(self) -> str:
        return os.getenv("LOG_LEVEL", "INFO")
    
    @property
    def MONITORING_DB(self) -> str:
        return "monitoring.db"
    
    @property
    def SUPABASE_URL(self) -> str:
        return os.getenv("SUPABASE_URL", "")
    
    @property
    def SUPABASE_KEY(self) -> str:
        return os.getenv("SUPABASE_KEY", "")
    
    def validate(self) -> bool:
        """Validate required configuration."""
        if not self.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required. Set it in .env file.")
        return True
    
    def is_using_supabase(self) -> bool:
        """Check if Supabase is configured."""
        return bool(self.SUPABASE_URL and self.SUPABASE_KEY)


# Global config instance
config = Config()
