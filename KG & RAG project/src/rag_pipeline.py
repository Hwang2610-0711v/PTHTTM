"""
Pipeline RAG: Load → Chunk → Embed → Store → Retrieve → Generate.
Sử dụng LangChain + ChromaDB + Sentence Transformers.
"""
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from config import (
    DATA_DIR, CHROMA_DIR, EMBEDDING_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K,
)
from src.llm_utils import generate_answer
from src.kg_builder import build_diabetes_kg, build_finance_kg, graph_to_text


# =========================================================
# 1. KHỞI TẠO EMBEDDING (chạy local, miễn phí)
# =========================================================
def get_embeddings():
    print(f"[RAG] Loading embedding model: {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# =========================================================
# 2. NẠP & CHUNK DỮ LIỆU
# =========================================================
def load_and_chunk(filepath: Path) -> list[Document]:
    loader = TextLoader(str(filepath), encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"[RAG] {filepath.name}: {len(chunks)} chunks")
    return chunks


# =========================================================
# 3. XÂY DỰNG VECTOR STORE
# =========================================================
def build_vectorstore(collection_name: str, source_files: list[Path],
                      extra_text: str = "") -> Chroma:
    """Tạo ChromaDB collection từ danh sách file + text bổ sung (từ KG)."""
    all_chunks = []
    for fp in source_files:
        all_chunks.extend(load_and_chunk(fp))

    # Thêm tri thức từ Knowledge Graph dưới dạng text
    if extra_text:
        kg_chunks = [
            Document(page_content=extra_text, metadata={"source": "knowledge_graph"})
        ]
        all_chunks.extend(kg_chunks)
        print(f"[RAG] Đã thêm {len(kg_chunks)} chunk từ Knowledge Graph")

    embeddings = get_embeddings()
    persist_dir = str(CHROMA_DIR / collection_name)

    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )
    print(f"[RAG] Đã lưu vector store: {persist_dir}")
    return vectorstore


def load_vectorstore(collection_name: str) -> Chroma:
    """Load lại vector store đã lưu."""
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=str(CHROMA_DIR / collection_name),
        embedding_function=embeddings,
        collection_name=collection_name,
    )


# =========================================================
# 4. TRUY XUẤT + SINH CÂU TRẢ LỜI
# =========================================================
def rag_query(vectorstore: Chroma, question: str, k: int = TOP_K) -> dict:
    """
    Thực hiện RAG: retrieve → generate.
    Trả về dict gồm: answer, contexts, sources.
    """
    # Retrieve
    docs = vectorstore.similarity_search(question, k=k)
    contexts = [d.page_content for d in docs]
    sources = [d.metadata.get("source", "unknown") for d in docs]

    # Ghép context
    joined_context = "\n\n---\n\n".join(contexts)

    # Generate
    answer = generate_answer(question, joined_context)

    return {
        "answer": answer,
        "contexts": contexts,
        "sources": sources,
    }


# =========================================================
# 5. KHỞI TẠO HỆ THỐNG
# =========================================================
def init_diabetes_system() -> Chroma:
    """Build vector store cho tiểu đường (gồm cả KG)."""
    kg = build_diabetes_kg()
    kg_text = "QUAN HỆ TỪ KNOWLEDGE GRAPH:\n" + graph_to_text(kg)
    return build_vectorstore(
        collection_name="diabetes",
        source_files=[DATA_DIR / "diabetes_knowledge.txt"],
        extra_text=kg_text,
    )


def init_finance_system() -> Chroma:
    """Build vector store cho tài chính (gồm cả KG)."""
    kg = build_finance_kg()
    kg_text = "QUAN HỆ TỪ KNOWLEDGE GRAPH:\n" + graph_to_text(kg)
    return build_vectorstore(
        collection_name="finance",
        source_files=[DATA_DIR / "finance_knowledge.txt"],
        extra_text=kg_text,
    )


# =========================================================
# 6. MAIN TEST
# =========================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TEST RAG PIPELINE")
    print("=" * 60)

    # --- Test Diabetes ---
    print("\n>>> Build Diabetes RAG...")
    vs_diab = init_diabetes_system()

    q1 = "Chỉ số Glucose bao nhiêu thì được coi là tiểu đường?"
    print(f"\nCâu hỏi: {q1}")
    result = rag_query(vs_diab, q1)
    print(f"Trả lời: {result['answer']}")
    print(f"Nguồn: {result['sources']}")

    # --- Test Finance ---
    print("\n>>> Build Finance RAG...")
    vs_fin = init_finance_system()

    q2 = "Khi lạm phát tăng thì giá vàng thế nào?"
    print(f"\nCâu hỏi: {q2}")
    result = rag_query(vs_fin, q2)
    print(f"Trả lời: {result['answer']}")
    print(f"Nguồn: {result['sources']}")