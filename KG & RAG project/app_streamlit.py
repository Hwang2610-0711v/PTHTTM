"""
Streamlit app demo 2 hệ thống: Tiểu đường & Tài chính.
Chạy: streamlit run app_streamlit.py
"""
import streamlit as st
from pathlib import Path

from config import CHROMA_DIR, KG_DIR
from src.rag_pipeline import (
    init_diabetes_system, init_finance_system,
    load_vectorstore, rag_query,
)

st.set_page_config(
    page_title="Chương 5: KG & RAG Demo",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Chương 5: Knowledge Graph & RAG Demo")
st.caption("Hệ thống Thông minh - Lê Thế Hoàng - B23DCCN331")


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("⚙️ Cấu hình")
    app_mode = st.radio("Chọn ứng dụng:", ["🩺 Tiểu đường", "💰 Tài chính (Vàng & Chứng khoán)"])

    st.divider()
    st.subheader("📊 Trạng thái")
    st.write(f"- Vector DB: `{CHROMA_DIR.name}/`")
    st.write(f"- KG Images: `{KG_DIR.name}/`")

    if st.button("🔄 Rebuild Vector Store"):
        with st.spinner("Đang build..."):
            if "Tiểu đường" in app_mode:
                init_diabetes_system()
            else:
                init_finance_system()
            st.success("Rebuild xong!")
            st.rerun()


# =========================================================
# LOAD VECTOR STORE (cache)
# =========================================================
@st.cache_resource
def get_vs(mode: str):
    collection = "diabetes" if mode == "diabetes" else "finance"
    persist = CHROMA_DIR / collection
    if not persist.exists():
        if mode == "diabetes":
            return init_diabetes_system()
        return init_finance_system()
    return load_vectorstore(collection)


# =========================================================
# MAIN TABS
# =========================================================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Hỏi đáp với RAG")

    if "Tiểu đường" in app_mode:
        mode_key = "diabetes"
        default_q = "Chỉ số Glucose bao nhiêu thì được coi là tiểu đường?"
        sample_questions = [
            "Chỉ số Glucose bao nhiêu thì được coi là tiểu đường?",
            "Yếu tố nguy cơ của bệnh tiểu đường là gì?",
            "Biến chứng của tiểu đường type 2 gồm những gì?",
            "Cách phòng ngừa bệnh tiểu đường?",
        ]
    else:
        mode_key = "finance"
        default_q = "Khi lạm phát tăng thì giá vàng thế nào?"
        sample_questions = [
            "Khi lạm phát tăng thì giá vàng thế nào?",
            "Lãi suất Fed ảnh hưởng gì đến chứng khoán?",
            "Chiến lược đầu tư an toàn là gì?",
            "Vàng và USD có quan hệ gì?",
        ]

    # Chọn câu hỏi mẫu
    st.write("**Câu hỏi gợi ý:**")
    cols = st.columns(2)
    for i, q in enumerate(sample_questions):
        if cols[i % 2].button(q, key=f"sample_{i}"):
            st.session_state["question"] = q

    # Ô nhập câu hỏi
    question = st.text_input("Nhập câu hỏi của bạn:", value=st.session_state.get("question", default_q))

    if st.button("🚀 Gửi câu hỏi", type="primary"):
        if not question.strip():
            st.warning("Vui lòng nhập câu hỏi.")
        else:
            with st.spinner("Đang truy xuất và sinh câu trả lời..."):
                vs = get_vs(mode_key)
                result = rag_query(vs, question)

            st.success("✅ Trả lời:")
            st.markdown(f"### {result['answer']}")

            with st.expander("📚 Ngữ cảnh đã truy xuất (Retrieved Contexts)"):
                for i, ctx in enumerate(result["contexts"], 1):
                    st.markdown(f"**Chunk {i}** (source: `{result['sources'][i-1]}`)")
                    st.text(ctx[:500] + ("..." if len(ctx) > 500 else ""))

with col2:
    st.subheader("🕸️ Knowledge Graph")
    if "Tiểu đường" in app_mode:
        kg_img = KG_DIR / "kg_diabetes.png"
    else:
        kg_img = KG_DIR / "kg_finance.png"

    if kg_img.exists():
        st.image(str(kg_img), caption="Đồ thị tri thức", use_container_width=True)
    else:
        st.info("Chưa có hình KG. Chạy `python -m src.kg_builder` trước.")