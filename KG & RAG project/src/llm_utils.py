"""
Wrapper gọi LLM. Hỗ trợ Gemini, OpenAI, và fallback template.
"""
from config import GEMINI_API_KEY, OPENAI_API_KEY, LLM_MODEL, TEMPERATURE


def get_llm():
    """Trả về đối tượng LLM (LangChain) hoặc None nếu không có API key."""
    if GEMINI_API_KEY:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=LLM_MODEL,
                google_api_key=GEMINI_API_KEY,
                temperature=TEMPERATURE,
            )
        except Exception as e:
            print(f"[LLM] Không khởi tạo được Gemini: {e}")

    if OPENAI_API_KEY:
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model="gpt-3.5-turbo",
                api_key=OPENAI_API_KEY,
                temperature=TEMPERATURE,
            )
        except Exception as e:
            print(f"[LLM] Không khởi tạo được OpenAI: {e}")

    print("[LLM] Không có API key → dùng chế độ FALLBACK template.")
    return None


def generate_answer(question: str, context: str) -> str:
    """
    Sinh câu trả lời dựa trên câu hỏi và ngữ cảnh retrieved.
    Nếu không có LLM → trả về template đơn giản.
    """
    prompt = f"""Bạn là trợ lý AI chuyên gia. Hãy trả lời câu hỏi dựa HOÀN TOÀN vào ngữ cảnh được cung cấp.
Nếu ngữ cảnh không đủ thông tin, hãy nói "Tôi không có đủ thông tin để trả lời".

NGỮ CẢNH:
{context}

CÂU HỎI: {question}

TRẢ LỜI (ngắn gọn, có căn cứ):"""

    llm = get_llm()    if llm is not None:
        try:
            response = llm.invoke(prompt)
            content = response.content
        
            # Xử lý trường hợp model mới trả về dạng list (chứa text và signature)
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                return "".join(text_parts)
            
            # Trường hợp model cũ trả về string bình thường
            return content
        
        except Exception as e:
            return f"[Lỗi gọi LLM: {e}]\n\nDựa trên ngữ cảnh retrieved:\n{context[:800]}..."

    # === FALLBACK: Không có API key ===
    return (
        "⚠️ Chưa cấu hình API key LLM. Dưới đây là ngữ cảnh liên quan nhất "
        "mà hệ thống RAG đã truy xuất được:\n\n"
        f"{context}\n\n"
        "👉 Để có câu trả lời thông minh, hãy thêm GEMINI_API_KEY vào file .env"
    )