import streamlit as st
from loaders.document_loader import load_document
from utils.chunking import split_documents
from embeddings.embedding_model import get_embeddings
from rag_pipeline.chatbot import create_rag_chain, ask_question
from langchain_community.vectorstores import FAISS
import tempfile
import os
import sys
print(sys.executable)

st.set_page_config(
    page_title="Dokument",
    page_icon=":)",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #1C1C1A;
    color: #D4CEBC;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] { display: none !important; }
.block-container { padding: 3rem 2rem; max-width: 760px; }

.app-title {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 400;
    color: #E8E2D0;
    letter-spacing: 0.02em;
    margin-bottom: 0.1rem;
}
.app-subtitle {
    font-size: 0.68rem;
    color: #524F4A;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}
.section-label {
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #524F4A;
    margin-bottom: 0.6rem;
}
.ready-badge {
    display: inline-block;
    background: #1E2A1A;
    border: 1px solid #3E4A38;
    color: #7A8C6E;
    font-size: 0.62rem;
    padding: 4px 12px;
    border-radius: 2px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.6rem;
    margin-bottom: 0.4rem;
}
.doc-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 0.4rem;
    margin-bottom: 1rem;
}
.doc-pill {
    background: transparent;
    border: 1px solid #2E2E2A;
    border-radius: 2px;
    padding: 3px 10px;
    font-size: 0.68rem;
    color: #6A6858;
}
.stButton > button {
    background: #2A2E26 !important;
    color: #9AAA8C !important;
    border: 1px solid #3A4234 !important;
    border-radius: 4px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.45rem 1.2rem !important;
    margin-top: 0.5rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #343D2E !important;
    border-color: #6A8060 !important;
    color: #C4D4B6 !important;
}
[data-testid="stFileUploader"] {
    background: #222220 !important;
    border: 1px dashed #38382E !important;
    border-radius: 6px !important;
}
[data-testid="stAlert"] { display: none !important; }
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.9rem 0 !important;
    border-bottom: 1px solid #222220 !important;
}
[data-testid="stChatInput"] {
    background: #222220 !important;
    border: 1px solid #2A2A26 !important;
    border-radius: 6px !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #4A5A44 !important;
}
[data-testid="stChatInput"] textarea {
    color: #D4CEBC !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 300 !important;
}
div[data-testid="stMarkdownContainer"] p {
    font-size: 0.9rem;
    line-height: 1.85;
    color: #C8C3B4;
    font-weight: 300;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──
if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "docs_list" not in st.session_state:
    st.session_state.docs_list = []
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0
if "indexed" not in st.session_state:
    st.session_state.indexed = False

# ── Header ──
st.markdown('<div class="app-title">Dokument</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">your personal document reader</div>', unsafe_allow_html=True)

# ── Upload section ──
st.markdown('<div class="section-label">Upload</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

if uploaded_files:
    if st.button("Index Documents"):
        embeddings = get_embeddings()
        combined_vectorstore = None
        total_chunks = 0
        new_docs = []

        for uploaded_file in uploaded_files:
            with st.spinner(f"Reading {uploaded_file.name}..."):
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=os.path.splitext(uploaded_file.name)[1]
                ) as f:
                    f.write(uploaded_file.read())
                    temp_path = f.name

                docs = load_document(temp_path)
                chunks = split_documents(docs)
                total_chunks += len(chunks)
                new_docs.append(uploaded_file.name)

                if combined_vectorstore is None:
                    combined_vectorstore = FAISS.from_documents(chunks, embeddings)
                else:
                    new_vs = FAISS.from_documents(chunks, embeddings)
                    combined_vectorstore.merge_from(new_vs)

                os.unlink(temp_path)

        st.session_state.chain = create_rag_chain(combined_vectorstore)
        st.session_state.docs_list = new_docs
        st.session_state.total_chunks = total_chunks
        st.session_state.messages = []
        st.session_state.indexed = True
        st.markdown(
            f'<div class="ready-badge">✓ ready · {total_chunks} chunks · {len(new_docs)} docs</div>',
            unsafe_allow_html=True
        )

if st.session_state.docs_list:
    pills = "".join([f'<span class="doc-pill">○ {d}</span>' for d in st.session_state.docs_list])
    st.markdown(f'<div class="doc-pills">{pills}</div>', unsafe_allow_html=True)

# ── Chat section ──
st.markdown('<div class="section-label">Conversation</div>', unsafe_allow_html=True)

if not st.session_state.messages and not st.session_state.indexed:
    st.markdown("""
    <div style="padding: 2rem 0; color: #2E2C28; font-size: 0.85rem;
                font-family: 'Playfair Display', serif; font-style: italic;">
        Index a document above to begin asking questions...
    </div>
    """, unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("ask something..."):
    if st.session_state.chain is None:
        st.warning("Please index a document first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner(""):
                answer, sources = ask_question(st.session_state.chain, prompt)
                st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})