import os
from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3-next-80b-a3b-instruct:free")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "300"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
EMBEDDING_DIMENSION = 768

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

TOP_K = int(os.getenv("TOP_K", "5"))

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "rag_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
