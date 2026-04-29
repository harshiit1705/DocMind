#  app.py

import streamlit as st
import os
import tempfile
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

# load_dotenv()
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f;
    color: #e8e6f0;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background: #0f0f1a !important;
    border-right: 1px solid #1e1e30;
}

[data-testid="stSidebar"] * { color: #e8e6f0 !important; }

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    letter-spacing: -0.02em;
}

/* Header */
.app-header {
    padding: 2rem 0 1rem 0;
    border-bottom: 1px solid #1e1e30;
    margin-bottom: 2rem;
}
.app-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.app-subtitle {
    color: #6b6b8a;
    font-size: 0.9rem;
    margin-top: 0.3rem;
    font-weight: 300;
}

/* Chat messages */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 1.2rem;
    padding: 1rem 0;
}
.msg-user {
    align-self: flex-end;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: #fff;
    padding: 0.85rem 1.2rem;
    border-radius: 18px 18px 4px 18px;
    max-width: 75%;
    font-size: 0.95rem;
    line-height: 1.5;
    box-shadow: 0 4px 20px rgba(79,70,229,0.3);
}
.msg-ai {
    align-self: flex-start;
    background: #13131f;
    border: 1px solid #1e1e30;
    color: #d4d2e8;
    padding: 0.85rem 1.2rem;
    border-radius: 18px 18px 18px 4px;
    max-width: 75%;
    font-size: 0.95rem;
    line-height: 1.6;
}
.msg-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
    opacity: 0.5;
}

/* Status badges */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.8rem;
    border-radius: 100px;
    font-size: 0.78rem;
    font-weight: 500;
    font-family: 'Syne', sans-serif;
}
.badge-success {
    background: rgba(52, 211, 153, 0.12);
    color: #34d399;
    border: 1px solid rgba(52,211,153,0.25);
}
.badge-warn {
    background: rgba(251, 191, 36, 0.12);
    color: #fbbf24;
    border: 1px solid rgba(251,191,36,0.25);
}
.badge-info {
    background: rgba(96, 165, 250, 0.12);
    color: #60a5fa;
    border: 1px solid rgba(96,165,250,0.25);
}

/* Chunk stats card */
.stats-card {
    background: #13131f;
    border: 1px solid #1e1e30;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
}
.stats-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.3rem 0;
    border-bottom: 1px solid #1a1a28;
    font-size: 0.85rem;
}
.stats-row:last-child { border-bottom: none; }
.stats-label { color: #6b6b8a; }
.stats-value { color: #a78bfa; font-weight: 600; font-family: 'Syne', sans-serif; }

/* Input */
[data-testid="stTextInput"] input {
    background: #13131f !important;
    border: 1px solid #1e1e30 !important;
    border-radius: 12px !important;
    color: #e8e6f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 0.75rem 1rem !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.2) !important;
}

/* Buttons */
.stButton button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    padding: 0.6rem 1.5rem !important;
    transition: opacity 0.2s !important;
}
.stButton button:hover { opacity: 0.85 !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: #13131f !important;
    border: 1.5px dashed #2a2a40 !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

/* Slider */
[data-testid="stSlider"] .st-emotion-cache-1inwz65 { color: #a78bfa !important; }

/* Spinner */
.stSpinner { color: #a78bfa !important; }

/* Divider */
hr { border-color: #1e1e30 !important; }

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: #3a3a58;
}
.empty-state-icon { font-size: 3rem; margin-bottom: 1rem; }
.empty-state-text {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
}
.empty-state-sub { font-size: 0.85rem; margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def process_pdf(uploaded_file, chunk_size, chunk_overlap):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    embedding_model = load_embedding_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=tmpdir,
        )
        # Re-load so it stays in memory (not tied to tmpdir lifecycle)
        vectorstore = Chroma(
            persist_directory=tmpdir,
            embedding_function=embedding_model,
        )
        # Keep a copy in memory
        st.session_state.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
        )

    os.unlink(tmp_path)
    return len(chunks)


def get_answer(query, k, lambda_mult):
    retriever = st.session_state.vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": k * 3, "lambda_mult": lambda_mult},
    )

    llm = init_chat_model("groq:llama-3.1-8b-instant")

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are DocMind, a helpful AI assistant.
Use ONLY the provided context to answer the question.
Be concise and precise. Format your answer clearly.
If the answer is not present in the context, say: "I couldn't find that in the document." """),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])

    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = prompt_template.invoke({"context": context, "question": query})
    response = llm.invoke(prompt)
    return response.content


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 DocMind AI")
    st.markdown("<div style='color:#6b6b8a;font-size:0.8rem;margin-bottom:1.5rem;'>Upload any PDF and chat with it</div>", unsafe_allow_html=True)

    st.markdown("**📄 Upload Document**")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        label_visibility="collapsed",
    )

    st.markdown("<div style='margin-top:1.2rem'><b>⚙️ Chunking Settings</b></div>", unsafe_allow_html=True)

    chunk_size = st.slider(
        "Chunk Size (chars)",
        min_value=200, max_value=2000,
        value=1000, step=100,
        help="How many characters per chunk"
    )
    chunk_overlap = st.slider(
        "Chunk Overlap (chars)",
        min_value=0, max_value=500,
        value=200, step=50,
        help="Overlap between consecutive chunks"
    )

    st.markdown("<div style='margin-top:1.2rem'><b>🔍 Retrieval Settings</b></div>", unsafe_allow_html=True)

    k_chunks = st.slider(
        "Chunks to Retrieve (k)",
        min_value=1, max_value=10,
        value=4,
        help="How many chunks to send as context"
    )
    lambda_mult = st.slider(
        "Diversity (λ)",
        min_value=0.0, max_value=1.0,
        value=0.5, step=0.1,
        help="0 = max diversity, 1 = max relevance"
    )

    if uploaded_file:
        if st.button("⚡ Process Document", use_container_width=True):
            with st.spinner("Processing PDF..."):
                try:
                    count = process_pdf(uploaded_file, chunk_size, chunk_overlap)
                    st.session_state.chunk_count = count
                    st.session_state.doc_name = uploaded_file.name
                    st.session_state.messages = []
                    st.success(f"Done! {count} chunks created.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    if st.session_state.doc_name:
        st.markdown("---")
        st.markdown(f"""
        <div class='stats-card'>
            <div class='stats-row'>
                <span class='stats-label'>Document</span>
                <span class='stats-value' style='font-size:0.75rem;max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>{st.session_state.doc_name}</span>
            </div>
            <div class='stats-row'>
                <span class='stats-label'>Total Chunks</span>
                <span class='stats-value'>{st.session_state.chunk_count}</span>
            </div>
            <div class='stats-row'>
                <span class='stats-label'>Chunk Size</span>
                <span class='stats-value'>{chunk_size} chars</span>
            </div>
            <div class='stats-row'>
                <span class='stats-label'>Overlap</span>
                <span class='stats-value'>{chunk_overlap} chars</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class='app-header'>
    <p class='app-title'>DocMind AI</p>
    <p class='app-subtitle'>RAG-powered document intelligence — ask anything about your PDF</p>
</div>
""", unsafe_allow_html=True)

# Status bar
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if st.session_state.vectorstore:
        st.markdown("<span class='badge badge-success'>● Document Ready</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='badge badge-warn'>○ No Document</span>", unsafe_allow_html=True)
with col2:
    st.markdown("<span class='badge badge-info'>⚡ Groq · LLaMA 3.1 8B</span>", unsafe_allow_html=True)

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

# Chat history display
chat_placeholder = st.container()

with chat_placeholder:
    if not st.session_state.messages:
        st.markdown("""
        <div class='empty-state'>
            <div class='empty-state-icon'>🧠</div>
            <div class='empty-state-text'>Upload a PDF to get started</div>
            <div class='empty-state-sub'>Then ask anything about your document</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class='msg-user'>
                    <div class='msg-label'>You</div>
                    {msg['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='msg-ai'>
                    <div class='msg-label'>DocMind</div>
                    {msg['content']}
                </div>
                """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# Input
if st.session_state.vectorstore:
    query = st.text_input(
        "Ask a question...",
        placeholder="e.g. What is gradient descent?",
        label_visibility="collapsed",
        key="user_input",
    )

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.spinner("Thinking..."):
            try:
                answer = get_answer(query, k_chunks, lambda_mult)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠️ Error: {str(e)}"
                })
        st.rerun()
else:
    st.text_input(
        "Ask a question...",
        placeholder="Upload and process a PDF first...",
        label_visibility="collapsed",
        disabled=True,
    )