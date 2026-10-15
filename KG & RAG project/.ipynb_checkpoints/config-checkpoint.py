{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "248a5136-5054-4be1-b747-8fdb0deea03c",
   "metadata": {},
   "outputs": [],
   "source": [
    "\"\"\"\n",
    "Cấu hình chung cho toàn bộ dự án\n",
    "\"\"\"\n",
    "import os\n",
    "from pathlib import Path\n",
    "from dotenv import load_dotenv\n",
    "\n",
    "load_dotenv()\n",
    "\n",
    "# === Đường dẫn ===\n",
    "BASE_DIR = Path(__file__).parent\n",
    "DATA_DIR = BASE_DIR / \"data\"\n",
    "CHROMA_DIR = BASE_DIR / \"chroma_db\"     # Nơi lưu Vector DB\n",
    "KG_DIR = BASE_DIR / \"kg_output\"          # Nơi lưu ảnh Knowledge Graph\n",
    "\n",
    "DATA_DIR.mkdir(exist_ok=True)\n",
    "KG_DIR.mkdir(exist_ok=True)\n",
    "CHROMA_DIR.mkdir(exist_ok=True)\n",
    "\n",
    "# === API Keys ===\n",
    "GEMINI_API_KEY = os.getenv(\"GEMINI_API_KEY\", \"\")\n",
    "OPENAI_API_KEY = os.getenv(\"OPENAI_API_KEY\", \"\")\n",
    "\n",
    "# === Embedding Model (chạy local, không cần API) ===\n",
    "EMBEDDING_MODEL = \"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2\"\n",
    "\n",
    "# === RAG Config ===\n",
    "CHUNK_SIZE = 500\n",
    "CHUNK_OVERLAP = 50\n",
    "TOP_K = 3   # Số lượng chunk truy xuất\n",
    "\n",
    "# === LLM Config ===\n",
    "LLM_MODEL = \"gemini-1.5-flash\"   # Hoặc \"gemini-1.5-pro\"\n",
    "TEMPERATURE = 0.3"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.15"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
