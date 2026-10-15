"""
Cấu hình chung cho toàn bộ dự án Chương 5.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# === Đường dẫn ===
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"     # Nơi lưu Vector DB
KG_DIR = BASE_DIR / "kg_output"          # Nơi lưu ảnh Knowledge Graph

DATA_DIR.mkdir(exist_ok=True)
KG_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

# === API Keys ===
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# === Embedding Model (chạy local, không cần API) ===
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# === RAG Config ===
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3   # Số lượng chunk truy xuất

# === LLM Config ===
LLM_MODEL = "gemini-3.8-flash"   # Hoặc "gemini-1.5-pro"
TEMPERATURE = 0.3