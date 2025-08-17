# StudyMate.ai - Full-Stack AI Study Application
# Complete implementation with PDF processing, embeddings, and LLM generation

import streamlit as st
import pandas as pd
import plotly.express as px
import time
import io
from typing import Dict, List, Optional

# Import our custom modules
from pdf_processor import PDFProcessor
from embeddings import EmbeddingsManager
from llm_generator import LLMGenerator
from quiz_manager import QuizManager
from config import Config

# -----------------------------
# Demo data
# -----------------------------
SAMPLE_QUESTIONS = [
    {
        "q": "What is the primary function of mitochondria in a eukaryotic cell?",
        "choices": [
            "Protein synthesis",
            "Waste disposal and recycling",
            "Cellular respiration and ATP production",
            "Storing genetic material",
        ],
        "answer_idx": 2,
        "explanation": (
            "Incorrect. While cells do have mechanisms for waste disposal (like lysosomes), "
            "this is not the primary function of mitochondria. Mitochondria are the powerhouses of the cell."
        ),
    },
    {
        "q": "Which organelle is responsible for protein synthesis?",
        "choices": ["Mitochondria", "Ribosome", "Lysosome", "Golgi apparatus"],
        "answer_idx": 1,
        "explanation": "Ribosomes synthesize proteins by translating mRNA.",
    },
    {
        "q": "Which molecule stores energy in cells for immediate use?",
        "choices": ["DNA", "ATP", "Cellulose", "Starch"],
        "answer_idx": 1,
        "explanation": "ATP (adenosine triphosphate) stores and transfers energy in cells.",
    },
]

TOPIC_PERFORMANCE = {"Topic 1": 5, "Topic 2": 9, "Topic 3": 4, "Topic 4": 8}

# -----------------------------
# Pixel-perfect CSS (light + dark)
# -----------------------------
PIXEL_CSS = {
    "light": """
    /* LIGHT THEME - Enhanced for complete light mode */
    :root{
      --bg: #f8fafc;
      --panel: #ffffff;
      --muted: #6b7280;
      --accent: #3b82f6;
      --accent-2: #6366f1;
      --soft: #f1f5f9;
      --danger: #fef2f2;
      --danger-border: #fecaca;
      --text: #1e293b;
      --subtle: #f8fafc;
      --border: #e2e8f0;
      --success: #10b981;
      --warning: #f59e0b;
      --error: #ef4444;
      --card-bg: #ffffff;
      --input-bg: #f9fafb;
      --button-bg: #f3f4f6;
      --button-hover: #e5e7eb;
    }
    html, body, [class*="css"], .stApp, .main .block-container {
      background: var(--bg) !important;
      color: var(--text) !important;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }

    /* Ensure sidebar follows theme */
    .css-1d391kg, .css-1lcbmhc, .css-1y4p8pa {
        background: var(--panel) !important;
        color: var(--text) !important;
    }

    /* Ensure all text elements follow theme */
    .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6, span, div {
        color: var(--text) !important;
    }

    /* File uploader styling for light theme */
    .stFileUploader > div {
        background: var(--panel) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
    }

    /* Fix file uploader drag area */
    .stFileUploader > div > div {
        background: var(--panel) !important;
        color: var(--text) !important;
        border: 2px dashed var(--border) !important;
        border-radius: 12px !important;
    }

    /* File uploader button */
    .stFileUploader button {
        background: var(--accent) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
    }

    /* Enhanced Radio button styling for light theme */
    .stRadio {
        background: transparent !important;
    }

    .stRadio > div {
        background: transparent !important;
        gap: 8px !important;
    }

    .stRadio > div > label {
        background: var(--panel) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        margin: 6px 0 !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
        font-weight: 500 !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
    }

    .stRadio > div > label:hover {
        background: var(--soft) !important;
        border-color: var(--accent) !important;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.15) !important;
    }

    /* Style the radio button circle */
    .stRadio input[type="radio"] {
        width: 16px !important;
        height: 16px !important;
        margin-right: 8px !important;
        accent-color: var(--accent) !important;
    }

    /* Selected radio button styling */
    .stRadio > div > label[data-checked="true"] {
        background: linear-gradient(135deg, rgba(59,130,246,0.1), rgba(99,102,241,0.1)) !important;
        border-color: var(--accent) !important;
        color: var(--accent) !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2) !important;
    }

    /* Text areas and inputs */
    .stTextArea > div > div > textarea,
    .stTextInput > div > div > input {
        background: var(--panel) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }
    /* hide default Streamlit header */
    .stApp > header { display: none; }

    /* app header */
    .app-header {
      display:flex; align-items:center; justify-content:space-between;
      padding:18px 22px; margin-bottom:20px;
      background: var(--panel);
      border-radius: 12px;
      box-shadow: 0 2px 8px rgba(15,23,42,0.04);
    }
    .logo { display:flex; align-items:center; gap:10px; font-weight:800; font-size:20px; color:#154c9c }
    .logo .star { background: linear-gradient(90deg,#e6f0ff,#dbeefe); padding:8px; border-radius:8px; box-shadow: 0 4px 10px rgba(47,128,237,0.08); }

    .top-tabs { display:flex; gap:10px; align-items:center }
    .tab { background:var(--panel); padding:8px 14px; border-radius:12px; font-size:14px; box-shadow: 0 2px 8px rgba(15,23,42,0.04); color: #394a6d }
    .tab.active { background: linear-gradient(90deg,var(--accent), var(--accent-2)); color: white }

    /* header right controls */
    .header-right { display:flex; gap:12px; align-items:center }
    .theme-toggle { background:var(--panel); border-radius:999px; padding:8px; box-shadow: 0 2px 8px rgba(15,23,42,0.04); cursor:pointer }

    /* layout - improved container styling */
    .container-card {
      background: var(--panel);
      border-radius:14px;
      padding:24px;
      margin-bottom: 16px;
      box-shadow: 0 8px 20px rgba(15,23,42,0.04);
      border: 1px solid rgba(15,23,42,0.08);
    }

    /* sidebar styling - improved */
    .sidebar .stFileUploader { width:100% }
    .upload-box {
      border:2px dashed var(--border) !important;
      border-radius:12px !important;
      padding:28px !important;
      text-align:center !important;
      color: var(--text-muted) !important;
      background: var(--panel) !important;
      transition: all 0.3s ease !important;
    }
    .upload-box:hover {
      border-color: var(--accent) !important;
      background: var(--soft) !important;
    }

    /* Force file uploader to be light */
    .stFileUploader, .stFileUploader * {
        background: var(--panel) !important;
        color: var(--text) !important;
    }

    /* File uploader drag area specific styling */
    div[data-testid="stFileUploader"] {
        background: var(--panel) !important;
    }

    div[data-testid="stFileUploader"] > div {
        background: var(--panel) !important;
        border: 2px dashed var(--border) !important;
        border-radius: 12px !important;
        color: var(--text) !important;
    }

    div[data-testid="stFileUploader"] label {
        color: var(--text) !important;
    }
    .pill {
      border-radius:999px;
      padding:10px 16px;
      border:1px solid #e9f0fb;
      margin-bottom:10px;
      display:inline-block;
      background:transparent;
      color: #4b5563;
      transition: all 0.2s ease;
    }
    .pill.active { background: #eef6ff; border:1px solid #d6e9ff; color: #1f6fe7 }

    .quick-notes {
        border-radius:12px !important;
        padding:16px !important;
        background: var(--soft) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
    }

    /* Fix any text areas that might be dark */
    .stTextArea textarea {
        background: var(--panel) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }

    /* main quiz styles */
    .progress-label { font-weight:600; color:#334155 }
    .progress-track { background:#e9eef9; border-radius:999px; padding:6px }
    .progress-bar { height:10px; border-radius:999px; background: linear-gradient(90deg,var(--accent), var(--accent-2)) }

    .question-card { background:var(--panel); border-radius:12px; padding:20px; }
    .choice { border:1px solid #e6edf6; padding:14px; border-radius:8px; margin-top:10px }
    .choice.correct { background:#eefef6; border-color:#bfe9d0 }
    .choice.wrong { background:var(--danger); border-color:var(--danger-border) }

    .explanation { background:#fff4f6; padding:12px; border-radius:8px; border:1px solid #f3c4c9; margin-top:12px; color:#7b1d24 }

    /* Results dark-card like the screenshot but in light mode we keep it slightly muted */
    .result-card { background: linear-gradient(180deg,#ffffff,#fbfcff); padding:18px; border-radius:12px }

    /* Button styling for light theme */
    .stButton>button {
        border-radius:12px;
        padding:12px 16px;
        font-weight: 500;
        background: var(--panel) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: var(--soft) !important;
        border-color: var(--accent) !important;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.15) !important;
    }

    /* Navigation button styling for light theme */
    div[data-testid="column"] .stButton>button {
        border-radius: 12px !important;
        border: 1px solid var(--border) !important;
        background: var(--panel) !important;
        color: var(--text) !important;
        font-weight: 500;
        transition: all 0.2s ease;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
    }
    div[data-testid="column"] .stButton>button:hover {
        border-color: var(--accent) !important;
        background: var(--soft) !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15) !important;
    }

    /* Primary button styling */
    .stButton>button[data-baseweb="button"][data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
    }

    /* Theme toggle button styling for light theme */
    button[data-testid="baseButton-secondary"] {
        background: var(--panel) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
    }

    /* Override Streamlit's default dark styling */
    .css-1d391kg, .css-1lcbmhc, .css-1y4p8pa, .css-12oz5g7, .css-1cpxqw2 {
        background: var(--bg) !important;
        color: var(--text) !important;
    }

    /* Fix any remaining dark elements */
    div[data-testid="stSidebar"] {
        background: var(--panel) !important;
        border-right: 1px solid var(--border) !important;
    }

    div[data-testid="stSidebar"] * {
        color: var(--text) !important;
    }

    /* Fix selectbox and other form elements */
    .stSelectbox > div > div {
        background: var(--panel) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
    }

    /* Fix any dark containers */
    .element-container, .stMarkdown, .stText {
        background: transparent !important;
        color: var(--text) !important;
    }

    /* Fix code blocks if any */
    .stCode {
        background: var(--soft) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
    }

    /* Fix any remaining white/light containers */
    .css-1kyxreq, .css-12oz5g7, .css-1cpxqw2, .css-1d391kg {
        background: var(--panel) !important;
        color: var(--text) !important;
    }

    /* Fix container and other cards */
    .stContainer, .css-1r6slb0, .css-1lcbmhc {
        background: var(--panel) !important;
        color: var(--text) !important;
    }

    /* Force all divs to follow theme */
    div[data-testid="stVerticalBlock"] > div {
        background: transparent !important;
    }

    /* Fix any white backgrounds */
    .css-1y4p8pa, .css-12oz5g7 {
        background: var(--bg) !important;
    }



    .section-title {
        font-weight: 700 !important;
        color: var(--text) !important;
        margin-bottom: 8px !important;
        font-size: 14px !important;
        letter-spacing: 0.5px !important;
    }

    """,
    "dark": """
    /* DARK THEME - Enhanced beautiful dark mode */
    :root{
      --bg: #0f172a;
      --panel: #1e293b;
      --muted: #94a3b8;
      --accent: #3b82f6;
      --accent-2: #6366f1;
      --soft: #1e293b;
      --danger: #1f2937;
      --danger-border: #374151;
      --text: #f1f5f9;
      --subtle: #334155;
      --success: #10b981;
      --warning: #f59e0b;
      --card-bg: #1e293b;
      --border: #334155;
    }
    html, body, [class*="css"]  {
      background: var(--bg) !important;
      color: var(--text) !important;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }
    .stApp > header { display: none; }

    .app-header {
      display:flex; align-items:center; justify-content:space-between;
      padding:18px 22px; margin-bottom:20px;
      background: var(--panel);
      border-radius: 12px;
      box-shadow: 0 4px 16px rgba(0,0,0,0.3);
      border: 1px solid var(--border);
    }
    .logo { display:flex; align-items:center; gap:10px; font-weight:800; font-size:20px; color:#60a5fa }
    .logo .star {
      background: linear-gradient(135deg, #3b82f6, #6366f1);
      padding:8px; border-radius:8px;
      box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }

    .top-tabs { display:flex; gap:10px; align-items:center }
    .tab {
      background:var(--panel); padding:10px 16px; border-radius:12px;
      font-size:14px; font-weight:500;
      box-shadow: 0 2px 8px rgba(0,0,0,0.2);
      color: var(--muted);
      border: 1px solid var(--border);
      transition: all 0.2s ease;
    }
    .tab.active {
      background: linear-gradient(135deg,var(--accent), var(--accent-2));
      color: white;
      box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3);
      border: 1px solid transparent;
    }

    .theme-toggle {
      background:var(--panel); border-radius:999px; padding:8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.2); cursor:pointer;
      border: 1px solid var(--border);
      transition: all 0.2s ease;
    }
    .theme-toggle:hover {
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }

    .container-card {
      background: var(--panel);
      border-radius:16px;
      padding:24px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.3);
      border: 1px solid var(--border);
    }

    .upload-box {
      border:2px dashed var(--border);
      border-radius:16px;
      padding:32px;
      text-align:center;
      color:var(--muted);
      background: linear-gradient(135deg, rgba(59,130,246,0.05), rgba(99,102,241,0.05));
      transition: all 0.3s ease;
    }
    .upload-box:hover {
      border-color: var(--accent);
      background: linear-gradient(135deg, rgba(59,130,246,0.1), rgba(99,102,241,0.1));
    }

    .pill {
      border-radius:999px;
      padding:12px 18px;
      border:1px solid var(--border);
      margin-bottom:12px;
      display:inline-block;
      color: var(--muted);
      background: var(--panel);
      transition: all 0.2s ease;
    }
    .pill.active {
      background: linear-gradient(135deg, rgba(59,130,246,0.2), rgba(99,102,241,0.2));
      border:1px solid rgba(59,130,246,0.4);
      color: var(--text);
      box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
    }

    .quick-notes {
      border-radius:12px;
      padding:16px;
      background: var(--soft);
      color: var(--text);
      border: 1px solid var(--border);
    }

    .progress-track {
      background: var(--soft);
      border-radius:999px;
      padding:4px;
      border: 1px solid var(--border);
    }
    .progress-bar {
      height:12px;
      border-radius:999px;
      background: linear-gradient(90deg,var(--accent), var(--accent-2));
      box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
    }

    .question-card {
      background:var(--panel);
      border-radius:16px;
      padding:24px;
      border: 1px solid var(--border);
      box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }

    .choice {
      border:1px solid var(--border);
      padding:16px;
      border-radius:12px;
      margin-top:12px;
      background: var(--soft);
      transition: all 0.2s ease;
      cursor: pointer;
    }
    .choice:hover {
      border-color: var(--accent);
      background: rgba(59,130,246,0.1);
    }
    .choice.correct {
      background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(16,185,129,0.1));
      border-color: var(--success);
      color: #6ee7b7;
    }
    .choice.wrong {
      background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(239,68,68,0.1));
      border-color: var(--error);
      color: #fca5a5;
    }

    .explanation {
      background: linear-gradient(135deg, rgba(59,130,246,0.1), rgba(99,102,241,0.1));
      padding:16px;
      border-radius:12px;
      border:1px solid rgba(59,130,246,0.3);
      margin-top:16px;
      color: var(--text);
    }

    .result-card {
      background: linear-gradient(135deg, var(--panel), var(--soft));
      padding:24px;
      border-radius:16px;
      border: 1px solid var(--border);
      box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

    .stButton>button {
      border-radius:12px;
      padding:12px 20px;
      font-weight: 500;
      transition: all 0.2s ease;
    }

    /* Navigation button styling for dark theme */
    div[data-testid="column"] .stButton>button {
        border-radius: 12px;
        border: 1px solid var(--border);
        background: var(--panel);
        color: var(--text);
        font-weight: 500;
        transition: all 0.2s ease;
    }
    div[data-testid="column"] .stButton>button:hover {
        border-color: var(--accent);
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3);
        background: rgba(59,130,246,0.1);
    }

    /* Theme toggle button styling */
    button[data-testid="baseButton-secondary"] {
        background: var(--panel) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
    }

    """,
}

# -----------------------------
# App logic / layout
# -----------------------------

st.set_page_config(page_title="StudyMate - Pixel", layout="wide")

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"
    if "page" not in st.session_state:
        st.session_state.page = "Normal"
    if "q_idx" not in st.session_state:
        st.session_state.q_idx = 0
    if "answers" not in st.session_state:
        st.session_state.answers = []

    # PDF and processing state
    if "pdf_uploaded" not in st.session_state:
        st.session_state.pdf_uploaded = False
    if "pdf_data" not in st.session_state:
        st.session_state.pdf_data = None
    if "processed_data" not in st.session_state:
        st.session_state.processed_data = None
    if "embeddings_ready" not in st.session_state:
        st.session_state.embeddings_ready = False

    # Quiz state
    if "quiz_generated" not in st.session_state:
        st.session_state.quiz_generated = False
    if "quiz_data" not in st.session_state:
        st.session_state.quiz_data = None
    if "quiz_started" not in st.session_state:
        st.session_state.quiz_started = False
    if "quiz_completed" not in st.session_state:
        st.session_state.quiz_completed = False

    # Summaries state
    if "page_summaries" not in st.session_state:
        st.session_state.page_summaries = []
    if "quick_notes" not in st.session_state:
        st.session_state.quick_notes = ""

    # Processing components
    if "pdf_processor" not in st.session_state:
        st.session_state.pdf_processor = PDFProcessor()
    if "embeddings_manager" not in st.session_state:
        st.session_state.embeddings_manager = EmbeddingsManager(Config.DEFAULT_EMBEDDING_MODEL)
    if "llm_generator" not in st.session_state:
        st.session_state.llm_generator = LLMGenerator(Config.DEFAULT_LLM_MODEL)
    if "quiz_manager" not in st.session_state:
        st.session_state.quiz_manager = QuizManager()

    # Progressive processing variables
    if "progressive_mode" not in st.session_state:
        st.session_state.progressive_mode = False
    if "processed_pages" not in st.session_state:
        st.session_state.processed_pages = {}
    if "pdf_data" not in st.session_state:
        st.session_state.pdf_data = None
    if "current_page_quiz" not in st.session_state:
        st.session_state.current_page_quiz = None

# Initialize session state
initialize_session_state()

# Helper functions
def toggle_theme():
    """Toggle between light and dark themes"""
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

def process_pdf_progressive(pdf_file):
    """Progressive PDF processing - load structure first, process pages on demand"""
    try:
        # Initialize processors
        pdf_processor = PDFProcessor(
            chunk_size=Config.PDF_CHUNK_SIZE,
            chunk_overlap=Config.PDF_CHUNK_OVERLAP
        )

        # Extract basic PDF info first
        st.info("📄 Extracting PDF structure...")
        pdf_data = pdf_processor.extract_text_from_pdf(pdf_file)

        if not pdf_data or not pdf_data.get('pages'):
            st.error("No text data extracted from PDF")
            return None

        # Store in session state for progressive processing
        st.session_state.pdf_data = pdf_data
        st.session_state.processed_pages = {}
        st.session_state.current_processing_page = 1
        st.session_state.total_pages = pdf_data['page_count']
        st.session_state.progressive_mode = True

        # Initialize managers (lightweight initialization)
        st.session_state.embeddings_manager = EmbeddingsManager(Config.DEFAULT_EMBEDDING_MODEL)
        st.session_state.llm_generator = LLMGenerator(Config.DEFAULT_LLM_MODEL)

        st.success(f"✅ PDF loaded successfully! {pdf_data['page_count']} pages detected.")
        st.info("📋 Pages will be processed individually as you navigate through them.")

        return {
            'pdf_data': pdf_data,
            'total_pages': pdf_data['page_count'],
            'ready_for_progressive': True
        }

    except Exception as e:
        st.error(f"Error in PDF loading: {str(e)}")
        return None

def process_single_page(page_number):
    """Process a single page with embeddings and summary"""
    try:
        if 'pdf_data' not in st.session_state:
            return None

        # Check if page already processed
        if page_number in st.session_state.processed_pages:
            return st.session_state.processed_pages[page_number]

        # Show processing status
        with st.spinner(f"🔄 Processing page {page_number}..."):
            pdf_data = st.session_state.pdf_data
            page_data = None

            # Find the page data
            for page in pdf_data['pages']:
                if page['page_number'] == page_number:
                    page_data = page
                    break

            if not page_data:
                st.error(f"Page {page_number} not found")
                return None

            # Create chunks for this page only
            pdf_processor = PDFProcessor()
            page_chunks = pdf_processor.create_text_chunks(
                page_data['text'],
                page_data['page_number']
            )

            # Create embeddings for this page
            embeddings_manager = st.session_state.embeddings_manager
            if page_chunks:
                try:
                    page_embeddings = embeddings_manager.create_embeddings(page_chunks)

                    # Build or update FAISS index
                    if not hasattr(embeddings_manager, 'index') or embeddings_manager.index is None:
                        embeddings_manager.build_faiss_index(page_embeddings, page_chunks)
                    else:
                        # Add to existing index
                        import faiss
                        import numpy as np
                        faiss.normalize_L2(page_embeddings)
                        embeddings_manager.index.add(page_embeddings.astype('float32'))
                        embeddings_manager.chunks_metadata.extend(page_chunks)
                except Exception as embed_error:
                    st.warning(f"Embedding creation failed for page {page_number}: {str(embed_error)}")

            # Generate summary for this page
            llm_generator = st.session_state.llm_generator
            try:
                summary = llm_generator.generate_page_summary(
                    page_data['text'],
                    page_data['page_number']
                )
            except Exception as summary_error:
                st.warning(f"Summary generation failed for page {page_number}: {str(summary_error)}")
                # Fallback to simple summary
                sentences = page_data['text'].split('.')[:3]
                summary = '. '.join([s.strip() for s in sentences if s.strip()]) + '.' if sentences else "Summary not available."

            # Store processed page data
            processed_page = {
                'page_number': page_number,
                'text': page_data['text'],
                'word_count': page_data['word_count'],
                'chunks': page_chunks,
                'summary': summary,
                'processed_at': time.time()
            }

            st.session_state.processed_pages[page_number] = processed_page

            return processed_page

    except Exception as e:
        st.error(f"Error processing page {page_number}: {str(e)}")
        return None

def generate_page_quiz(page_number, num_questions=5):
    """Generate quiz for a specific page"""
    try:
        if page_number not in st.session_state.processed_pages:
            # Process the page first
            process_single_page(page_number)

        if page_number not in st.session_state.processed_pages:
            st.error(f"Could not process page {page_number}")
            return None

        page_data = st.session_state.processed_pages[page_number]
        llm_generator = st.session_state.llm_generator

        with st.spinner(f"🧠 Generating quiz for page {page_number}..."):
            # Always use API for better quality questions
            questions = llm_generator.generate_quiz_questions(
                page_data['text'],
                num_questions,
                use_api=True
            )

            # Create quiz
            quiz_manager = QuizManager()
            quiz_title = f"Page {page_number} Quiz"
            quiz_data = quiz_manager.create_quiz(questions, quiz_title)

            return {
                'quiz_manager': quiz_manager,
                'quiz_data': quiz_data,
                'page_number': page_number
            }

    except Exception as e:
        st.error(f"Error generating quiz for page {page_number}: {str(e)}")
        return None

def process_uploaded_pdf(uploaded_file):
    """Process uploaded PDF and generate embeddings with comprehensive error handling"""
    try:
        # Validate file
        if uploaded_file is None:
            st.error("No file uploaded. Please select a PDF file.")
            return False

        if uploaded_file.size == 0:
            st.error("The uploaded file is empty. Please select a valid PDF file.")
            return False

        if uploaded_file.size > 50 * 1024 * 1024:  # 50MB limit
            st.error("File too large. Please upload a PDF smaller than 50MB.")
            return False

        # Reset uploaded file pointer and create a copy for processing
        uploaded_file.seek(0)

        # Create a copy of the file content to avoid issues with multiple reads
        file_content = uploaded_file.read()
        uploaded_file.seek(0)  # Reset for any other operations

        # Create a new file-like object from the content
        from io import BytesIO
        pdf_file_copy = BytesIO(file_content)
        pdf_file_copy.name = uploaded_file.name

        with st.spinner("Processing PDF..."):
            try:
                # Add debugging information
                st.info(f"Processing file: {uploaded_file.name} ({uploaded_file.size} bytes)")

                # Process PDF using the copy
                processed_data = st.session_state.pdf_processor.process_pdf_for_embeddings(pdf_file_copy)

                if not processed_data:
                    st.error("PDF processing returned no data. The file might be corrupted or empty.")
                    return False

                if not processed_data.get('chunks'):
                    st.error("Could not extract text chunks from PDF. The document might contain only images or be password protected.")
                    return False

                # Show processing results
                metadata = processed_data.get('pdf_metadata', {})
                st.success(f"✅ Successfully processed {metadata.get('page_count', 0)} pages with {len(processed_data['chunks'])} text chunks")

                st.session_state.processed_data = processed_data
                st.session_state.pdf_uploaded = True

            except Exception as pdf_error:
                st.error(f"PDF processing failed: {str(pdf_error)}")
                # Add more detailed error information
                import traceback
                st.error("Detailed error information:")
                st.code(traceback.format_exc())
                return False

            # Generate embeddings
            with st.spinner("Creating embeddings..."):
                try:
                    chunks = processed_data['chunks']
                    if not chunks:
                        st.error("No text chunks found in PDF. The document might be empty or contain only images.")
                        return False

                    st.info(f"Creating embeddings for {len(chunks)} text chunks...")
                    embeddings = st.session_state.embeddings_manager.create_embeddings(chunks)

                    st.info("Building search index...")
                    st.session_state.embeddings_manager.build_faiss_index(embeddings, chunks)
                    st.session_state.embeddings_ready = True

                    st.success(f"✅ Created embeddings and search index for {len(chunks)} chunks")

                except Exception as embedding_error:
                    st.error(f"Embedding generation failed: {str(embedding_error)}")
                    st.warning("Continuing without embeddings. Q&A functionality will be limited.")
                    st.session_state.embeddings_ready = False
                    # Don't return False here - continue with PDF processing

            # Generate page summaries
            with st.spinner("Generating summaries..."):
                try:
                    summaries = []
                    for page in processed_data['pages']:
                        try:
                            summary = st.session_state.llm_generator.generate_page_summary(
                                page['text'], page['page_number']
                            )
                            summaries.append({
                                'page_number': page['page_number'],
                                'summary': summary if summary else "Summary generation failed for this page.",
                                'word_count': page['word_count']
                            })
                        except Exception as page_error:
                            st.warning(f"Failed to generate summary for page {page['page_number']}: {str(page_error)}")
                            summaries.append({
                                'page_number': page['page_number'],
                                'summary': f"Page {page['page_number']} content (summary generation failed)",
                                'word_count': page['word_count']
                            })

                    st.session_state.page_summaries = summaries

                except Exception as summary_error:
                    st.error(f"Summary generation failed: {str(summary_error)}")
                    # Create basic summaries from raw text
                    basic_summaries = []
                    for page in processed_data['pages']:
                        basic_summaries.append({
                            'page_number': page['page_number'],
                            'summary': page['text'][:200] + "..." if len(page['text']) > 200 else page['text'],
                            'word_count': page['word_count']
                        })
                    st.session_state.page_summaries = basic_summaries
                    st.warning("Using basic text excerpts instead of AI-generated summaries.")

            # Generate QuickNotes using API
            with st.spinner("Creating QuickNotes..."):
                try:
                    # Use Gemini API for QuickNotes with learning mode
                    if Config.GOOGLE_API_KEY:
                        current_learning_mode = getattr(st.session_state, 'learning_mode', 'Intermediate')
                        quick_notes = st.session_state.llm_generator._generate_quick_notes_with_gemini(
                            processed_data['total_text'],
                            current_learning_mode
                        )
                        if quick_notes and quick_notes.strip():
                            st.session_state.quick_notes = quick_notes
                            # Also store with learning mode key for better organization
                            setattr(st.session_state, f'quick_notes_{current_learning_mode}', quick_notes)
                            logger.info(f"Stored QuickNotes for {current_learning_mode} mode: {len(quick_notes)} characters")
                        else:
                            st.session_state.quick_notes = "QuickNotes generation failed - API returned empty response."
                            logger.warning("QuickNotes generation returned empty response")
                    else:
                        # Fallback to local generation with learning mode
                        current_learning_mode = getattr(st.session_state, 'learning_mode', 'Intermediate')
                        quick_notes = st.session_state.llm_generator.generate_quick_notes(
                            processed_data['total_text'],
                            current_learning_mode
                        )
                        st.session_state.quick_notes = quick_notes if quick_notes else "QuickNotes generation failed."

                except Exception as notes_error:
                    st.warning(f"QuickNotes generation failed: {str(notes_error)}")
                    # Create basic notes from first few sentences
                    sentences = processed_data['total_text'].split('.')[:5]
                    st.session_state.quick_notes = '\n'.join([f"• {sentence.strip()}" for sentence in sentences if sentence.strip()])

            st.success("PDF processed successfully!")
            return True

    except Exception as e:
        st.error(f"Unexpected error during PDF processing: {str(e)}")
        st.error("Please try uploading a different PDF file or contact support if the issue persists.")
        return False

def generate_quiz():
    """Generate quiz questions from processed PDF with error handling"""
    try:
        if not st.session_state.processed_data:
            st.error("Please upload and process a PDF first.")
            return False

        if not st.session_state.processed_data.get('total_text'):
            st.error("No text content found in the processed PDF.")
            return False

        text_length = len(st.session_state.processed_data['total_text'])
        if text_length < 100:
            st.error("PDF content is too short to generate meaningful quiz questions.")
            return False

        with st.spinner("🎯 Generating quiz questions..."):
            try:
                # Always use API for better quality questions (actually Gemini)
                questions = st.session_state.llm_generator.generate_quiz_questions(
                    st.session_state.processed_data['total_text'],
                    num_questions=10,
                    use_api=True
                )

                if not questions or len(questions) == 0:
                    st.error("Failed to generate quiz questions. The content might not be suitable for quiz generation.")
                    return False

                # Show success message
                st.success("✅ **Quiz generated successfully!** 🚀")

                # Validate questions
                valid_questions = []
                for q in questions:
                    if (q.get('q') and q.get('choices') and
                        len(q.get('choices', [])) >= 2 and
                        'answer_idx' in q):
                        valid_questions.append(q)

                if len(valid_questions) == 0:
                    st.error("No valid quiz questions could be generated from the content.")
                    return False

                quiz_data = st.session_state.quiz_manager.create_quiz(
                    valid_questions,
                    f"Quiz: {st.session_state.processed_data['pdf_metadata']['filename']}"
                )

                st.session_state.quiz_data = quiz_data
                st.session_state.quiz_generated = True

                if len(valid_questions) < 10:
                    st.warning(f"Only {len(valid_questions)} valid questions were generated instead of 10.")

            except Exception as quiz_error:
                st.error(f"Quiz generation failed: {str(quiz_error)}")
                return False

        st.success(f"Quiz generated successfully with {len(valid_questions)} questions!")
        return True

    except Exception as e:
        st.error(f"Unexpected error during quiz generation: {str(e)}")
        return False

def restart_quiz():
    """Restart the current quiz"""
    st.session_state.quiz_manager.reset_quiz()
    st.session_state.quiz_started = False
    st.session_state.quiz_completed = False
    st.session_state.q_idx = 0
    st.session_state.answers = []

# inject CSS
css = PIXEL_CSS[st.session_state.theme]
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# header with theme toggle
header_col1, header_col2 = st.columns([0.85, 0.15])

with header_col1:
    st.markdown(
        f"""
        <div class='app-header'>
            <div style='display:flex;align-items:center;gap:12px'>
                <div class="logo">
                    <div class="star">✨</div>
                    STUDY<span style='font-weight:700'>MATE.ai</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with header_col2:
    st.markdown("<div style='margin-top:10px'>", unsafe_allow_html=True)
    theme_icon = "🌞" if st.session_state.theme == "dark" else "🌙"
    theme_label = "Light Mode" if st.session_state.theme == "dark" else "Dark Mode"

    if st.button(f"{theme_icon}", help=f"Switch to {theme_label}", key="theme_toggle"):
        toggle_theme()
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# Navigation tabs
nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1,1,1,1])

with nav_col1:
    if st.button("📚 Normal", key="nav_normal_header", use_container_width=True,
                 type="primary" if st.session_state.page == "Normal" else "secondary"):
        st.session_state.page = "Normal"
        st.rerun()

with nav_col2:
    if st.button("📝 QuickNotes", key="nav_quicknotes_header", use_container_width=True,
                 type="primary" if st.session_state.page == "QuickNotes" else "secondary"):
        st.session_state.page = "QuickNotes"
        st.rerun()

with nav_col3:
    if st.button("🧠 Quiz", key="nav_quiz_header", use_container_width=True,
                 type="primary" if st.session_state.page == "Quiz" else "secondary"):
        st.session_state.page = "Quiz"
        st.rerun()

with nav_col4:
    if st.button("📊 Result", key="nav_result_header", use_container_width=True,
                 type="primary" if st.session_state.page == "Result" else "secondary"):
        st.session_state.page = "Result"
        st.rerun()

# Add some spacing
st.markdown("<br>", unsafe_allow_html=True)

# columns: sidebar + main
sidebar, main = st.columns([0.3, 0.7], gap="medium")

with sidebar:
    st.markdown("<div class='container-card sidebar'>", unsafe_allow_html=True)

    # PDF Upload Section
    st.markdown("<div style='font-weight:700;color:#3b82f6;margin-bottom:8px'>UPLOAD PDF</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF document to start studying"
    )

    if uploaded_file is not None and not st.session_state.pdf_uploaded:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Quick Process (Progressive)", type="primary"):
                # Create a new file-like object from the content
                from io import BytesIO
                file_content = uploaded_file.read()
                pdf_file_copy = BytesIO(file_content)
                pdf_file_copy.name = uploaded_file.name

                result = process_pdf_progressive(pdf_file_copy)
                if result:
                    st.session_state.pdf_uploaded = True
                    st.session_state.progressive_mode = True
                    st.rerun()

        with col2:
            if st.button("📄 Full Process (Traditional)", type="secondary"):
                success = process_uploaded_pdf(uploaded_file)
                if success:
                    st.session_state.progressive_mode = False
                    st.rerun()

    # Show PDF status
    if st.session_state.pdf_uploaded:
        st.success("✅ PDF processed successfully!")
        if st.session_state.processed_data:
            metadata = st.session_state.processed_data['pdf_metadata']
            st.info(f"📄 {metadata['filename']}\n📊 {metadata['page_count']} pages, {metadata['total_word_count']} words")

    st.write("\n")

    # Learning Mode Section
    st.markdown("<div style='font-weight:700;color:#334155;margin-bottom:8px'>LEARNING MODE</div>", unsafe_allow_html=True)
    learning_mode = st.radio(
        "Select your learning level",
        ("Beginner", "Intermediate", "Advanced"),
        index=0,
        label_visibility="collapsed"
    )

    # Store learning mode in session state
    st.session_state.learning_mode = learning_mode

    st.write("\n")

    # Quiz Generation Section
    if st.session_state.pdf_uploaded and not st.session_state.quiz_generated:
        if st.button("🧠 Generate Quiz", type="secondary"):
            success = generate_quiz()
            if success:
                st.rerun()



    st.markdown("</div>", unsafe_allow_html=True)

with main:

    # Pages
    if st.session_state.page == "Normal":
        st.markdown("<div class='container-card'>", unsafe_allow_html=True)

        if not st.session_state.pdf_uploaded:
            st.title("📚 Welcome to StudyMate.ai")
            st.write("Upload a PDF document to get started with AI-powered studying!")


            st.markdown("### 🎯 **Adaptive Learning System**")
            st.markdown("""
            **📚 Beginner**: Simple language, basic concepts, easy-to-understand explanations

            **🎓 Intermediate**: Balanced technical terms, practical applications, moderate detail

            **🔬 Advanced**: Technical terminology, in-depth analysis, complex relationships
            """)

            st.info("👈 Use the sidebar to upload your PDF file and select your learning level")
        else:
            # Check if we're in progressive mode
            if getattr(st.session_state, 'progressive_mode', False):
                st.title("📄 Progressive Page Processing")

                if 'pdf_data' in st.session_state:
                    total_pages = st.session_state.pdf_data['page_count']

                    # Page selector
                    st.subheader("📖 Select Page to Process")
                    selected_page = st.selectbox(
                        "Choose a page:",
                        range(1, total_pages + 1),
                        format_func=lambda x: f"Page {x}",
                        key="page_selector"
                    )

                    col1, col2, col3 = st.columns([1, 1, 1])

                    with col1:
                        if st.button("🔄 Process This Page", type="primary"):
                            process_single_page(selected_page)
                            st.rerun()

                    with col2:
                        if selected_page in st.session_state.processed_pages:
                            if st.button("🧠 Generate Page Quiz", type="secondary"):
                                quiz_result = generate_page_quiz(selected_page, 5)
                                if quiz_result:
                                    st.session_state.quiz_manager = quiz_result['quiz_manager']
                                    st.session_state.quiz_data = quiz_result['quiz_data']
                                    st.session_state.quiz_generated = True
                                    st.session_state.current_page_quiz = selected_page
                                    st.success(f"Quiz generated for page {selected_page}!")

                    with col3:
                        if st.session_state.quiz_generated and hasattr(st.session_state, 'current_page_quiz'):
                            if st.button("🚀 Start Page Quiz", type="primary"):
                                st.session_state.page = "Quiz"
                                st.rerun()

                    # Show page content if processed
                    if selected_page in st.session_state.processed_pages:
                        page_data = st.session_state.processed_pages[selected_page]

                        st.write("---")
                        with st.expander(f"📖 Page {selected_page} Summary ({page_data['word_count']} words)", expanded=True):
                            st.write(page_data['summary'])

                            # Q&A section for this page
                            st.write("---")
                            st.subheader("💬 Ask a Question")
                            question = st.text_input(
                                f"Ask about Page {selected_page}:",
                                key=f"question_page_{selected_page}",
                                placeholder="What would you like to know about this page?"
                            )

                            if question and st.button(f"Get Answer", key=f"answer_page_{selected_page}"):
                                if not question.strip():
                                    st.warning("Please enter a valid question.")
                                else:
                                    try:
                                        with st.spinner("Finding relevant information..."):
                                            # Use page-specific context
                                            context = page_data['text'][:1000]  # Use first 1000 chars as context
                                            answer = st.session_state.llm_generator.answer_question(question, context)
                                            if answer and answer.strip():
                                                st.markdown("### 💡 Answer:")
                                                st.text_area("", value=answer, height=150, disabled=True, label_visibility="collapsed")
                                            else:
                                                st.warning("Could not generate an answer. Please try rephrasing your question.")
                                    except Exception as qa_error:
                                        st.error(f"Error answering question: {str(qa_error)}")
                    else:
                        st.info(f"Page {selected_page} not processed yet. Click 'Process This Page' to analyze it.")

                    # Show processing progress
                    processed_count = len(st.session_state.processed_pages)
                    st.write("---")
                    st.subheader("📊 Processing Progress")
                    progress_pct = (processed_count / total_pages) * 100
                    st.progress(progress_pct / 100)
                    st.write(f"Processed: {processed_count}/{total_pages} pages ({progress_pct:.1f}%)")

            else:
                # Traditional mode - show all summaries
                st.title("📄 Document Analysis")

                # General Summary Section
                if st.session_state.processed_data and 'total_text' in st.session_state.processed_data:
                    st.subheader("📋 General Summary")

                    # Generate general summary if not already done or if learning mode changed
                    current_learning_mode = getattr(st.session_state, 'learning_mode', 'Intermediate')
                    summary_key = f'general_summary_{current_learning_mode}'

                    if not hasattr(st.session_state, summary_key):
                        with st.spinner(f"Generating document overview for {current_learning_mode} level..."):
                            try:
                                # Use Gemini API for general summary with learning mode
                                if Config.GOOGLE_API_KEY:
                                    summary = st.session_state.llm_generator._generate_general_summary_with_gemini(
                                        st.session_state.processed_data['total_text'],
                                        current_learning_mode
                                    )
                                    if summary:
                                        setattr(st.session_state, summary_key, summary)
                                    else:
                                        setattr(st.session_state, summary_key, "Unable to generate general summary at this time.")
                                else:
                                    setattr(st.session_state, summary_key, "API key required for general summary generation.")
                            except Exception as e:
                                setattr(st.session_state, summary_key, "Unable to generate general summary at this time.")
                                logger.error(f"Error generating general summary: {e}")

                    # Display general summary with learning mode indicator
                    if hasattr(st.session_state, summary_key):
                        summary_content = getattr(st.session_state, summary_key)

                        # Add learning mode indicator
                        mode_colors = {"Beginner": "🟢", "Intermediate": "🟡", "Advanced": "🔴"}
                        mode_color = mode_colors.get(current_learning_mode, "🟡")
                        st.markdown(f"**{mode_color} {current_learning_mode} Level Summary**")

                        st.text_area("", value=summary_content, height=200, disabled=True, label_visibility="collapsed")
                    else:
                        st.text_area("", value=f"Generating {current_learning_mode} level summary... Please wait or refresh the page.", height=200, disabled=True, label_visibility="collapsed")

                    st.write("---")

                st.subheader("📖 Page-by-Page Analysis")

                if st.session_state.page_summaries:
                    for summary_data in st.session_state.page_summaries:
                        with st.expander(f"📖 Page {summary_data['page_number']} Summary ({summary_data['word_count']} words)"):
                            st.write(summary_data['summary'])

                            # Add Q&A section for each page
                            st.write("---")
                            st.subheader("💬 Ask a Question")
                            question = st.text_input(
                                f"Ask about Page {summary_data['page_number']}:",
                                key=f"question_page_{summary_data['page_number']}",
                                placeholder="What would you like to know about this page?"
                            )

                            if question and st.button(f"Get Answer", key=f"answer_page_{summary_data['page_number']}"):
                                if not question.strip():
                                    st.warning("Please enter a valid question.")
                                elif st.session_state.embeddings_ready:
                                    try:
                                        with st.spinner("Finding relevant information..."):
                                            context = st.session_state.embeddings_manager.get_context_for_question(question)
                                            if not context:
                                                st.warning("No relevant context found for your question. Try rephrasing it.")
                                            else:
                                                answer = st.session_state.llm_generator.answer_question(question, context)
                                                if answer and answer.strip():
                                                    st.markdown("### 💡 Answer:")
                                                    st.text_area("", value=answer, height=150, disabled=True, label_visibility="collapsed")
                                                else:
                                                    st.warning("Could not generate an answer. Please try rephrasing your question.")
                                    except Exception as qa_error:
                                        st.error(f"Error answering question: {str(qa_error)}")
                                else:
                                    st.warning("Embeddings not ready. Please wait for processing to complete or try again later.")

                    if st.session_state.quiz_generated:
                        st.write("---")
                        if st.button("🧠 Start Quiz →", type="primary"):
                            st.session_state.page = "Quiz"
                            st.rerun()
                    elif st.session_state.pdf_uploaded:
                        st.write("---")
                        if st.button("🧠 Generate Quiz →", type="secondary"):
                            success = generate_quiz()
                            if success:
                                st.rerun()
                else:
                    st.info("Processing summaries... Please wait.")

        st.markdown("</div>", unsafe_allow_html=True)

    elif st.session_state.page == "QuickNotes":
        st.markdown("<div class='container-card'>", unsafe_allow_html=True)

        if not st.session_state.pdf_uploaded:
            st.title("📝 QuickNotes")
            st.info("Upload a PDF to generate condensed study notes!")
        else:
            st.title("📝 QuickNotes")

            # Check if QuickNotes need regeneration for current learning mode
            current_learning_mode = getattr(st.session_state, 'learning_mode', 'Intermediate')
            notes_key = f'quick_notes_{current_learning_mode}'

            if hasattr(st.session_state, notes_key):
                # Display existing notes for current learning mode
                mode_colors = {"Beginner": "🟢", "Intermediate": "🟡", "Advanced": "🔴"}
                mode_color = mode_colors.get(current_learning_mode, "🟡")
                st.markdown(f"### 🎯 Key Concepts - {mode_color} {current_learning_mode} Level")

                notes_content = getattr(st.session_state, notes_key)
                st.markdown(f"<div class='quick-notes'>{notes_content}</div>", unsafe_allow_html=True)

                # Add regenerate and download buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"🔄 Regenerate for {current_learning_mode} Level"):
                        delattr(st.session_state, notes_key)
                        st.rerun()

                with col2:
                    import datetime
                    import re
                    from pdf_generator import pdf_generator

                    # Clean notes content for text download
                    clean_notes = re.sub('<[^<]+?>', '', notes_content)
                    clean_notes = clean_notes.replace('&nbsp;', ' ').replace('&amp;', '&')

                    st.download_button(
                        label="📄 Download TXT",
                        data=clean_notes,
                        file_name=f"quicknotes_{current_learning_mode.lower()}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )

                    # Generate PDF for download
                    try:
                        pdf_bytes = pdf_generator.generate_quicknotes_pdf(
                            notes_content,
                            current_learning_mode,
                            "QuickNotes"
                        )
                        st.download_button(
                            label="📕 Download PDF",
                            data=pdf_bytes,
                            file_name=f"quicknotes_{current_learning_mode.lower()}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                    except Exception as e:
                        st.error(f"PDF generation failed: {str(e)}")

            elif st.session_state.quick_notes and st.session_state.quick_notes.strip():
                # Legacy notes exist, show them but offer to regenerate for current mode
                st.markdown("### 🎯 Key Concepts")
                st.markdown(f"<div class='quick-notes'>{st.session_state.quick_notes}</div>", unsafe_allow_html=True)

                # Add buttons for regenerate and download
                col1, col2 = st.columns(2)
                with col1:
                    regenerate_btn = st.button(f"🎯 Generate {current_learning_mode} Level Notes")

                with col2:
                    import datetime
                    import re
                    from pdf_generator import pdf_generator

                    # Clean notes content for text download
                    clean_notes = re.sub('<[^<]+?>', '', st.session_state.quick_notes)
                    clean_notes = clean_notes.replace('&nbsp;', ' ').replace('&amp;', '&')

                    st.download_button(
                        label="📄 Download TXT",
                        data=clean_notes,
                        file_name=f"quicknotes_general_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )

                    # Generate PDF for download
                    try:
                        pdf_bytes = pdf_generator.generate_quicknotes_pdf(
                            st.session_state.quick_notes,
                            current_learning_mode,
                            "QuickNotes"
                        )
                        st.download_button(
                            label="📕 Download PDF",
                            data=pdf_bytes,
                            file_name=f"quicknotes_general_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                    except Exception as e:
                        st.error(f"PDF generation failed: {str(e)}")

                if regenerate_btn:
                    # Generate new notes for current learning mode
                    with st.spinner(f"Generating {current_learning_mode} level notes..."):
                        try:
                            if Config.GOOGLE_API_KEY:
                                new_notes = st.session_state.llm_generator._generate_quick_notes_with_gemini(
                                    st.session_state.processed_data['total_text'],
                                    current_learning_mode
                                )
                            else:
                                new_notes = st.session_state.llm_generator.generate_quick_notes(
                                    st.session_state.processed_data['total_text'],
                                    current_learning_mode
                                )

                            if new_notes:
                                setattr(st.session_state, notes_key, new_notes)
                                st.rerun()
                        except Exception as e:
                            st.error(f"Failed to generate {current_learning_mode} level notes: {str(e)}")

                # Add Q&A section for QuickNotes
                st.write("---")
                st.subheader("💬 Ask About Key Concepts")
                question = st.text_input(
                    "Ask about the key concepts:",
                    key="quicknotes_question",
                    placeholder="What would you like to know more about?"
                )

                if question and st.button("Get Answer", key="quicknotes_answer"):
                    if not question.strip():
                        st.warning("Please enter a valid question.")
                    elif st.session_state.embeddings_ready:
                        try:
                            with st.spinner("Searching for relevant information..."):
                                context = st.session_state.embeddings_manager.get_context_for_question(question)
                                if not context:
                                    st.warning("No relevant context found. Try asking about the main concepts in the document.")
                                else:
                                    answer = st.session_state.llm_generator.answer_question(question, context)
                                    if answer and answer.strip():
                                        st.markdown("### 💡 Answer:")
                                        st.text_area("", value=answer, height=150, disabled=True, label_visibility="collapsed")
                                    else:
                                        st.warning("Could not generate an answer. Please try rephrasing your question.")
                        except Exception as qa_error:
                            st.error(f"Error answering question: {str(qa_error)}")
                    else:
                        st.warning("Embeddings not ready. Please wait for processing to complete.")

                # Personal notes section
                st.write("---")
                st.subheader("✏️ Your Personal Notes")
                personal_notes = st.text_area(
                    "Add your own notes and insights:",
                    height=150,
                    placeholder="Write your thoughts, questions, or additional insights here...",
                    key="personal_notes_input"
                )

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("💾 Save Notes"):
                        if personal_notes.strip():
                            # Store notes in session state
                            if 'saved_personal_notes' not in st.session_state:
                                st.session_state.saved_personal_notes = []

                            # Add timestamp and save
                            import datetime
                            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            note_entry = f"[{timestamp}]\n{personal_notes}\n\n"
                            st.session_state.saved_personal_notes.append(note_entry)
                            st.success("Notes saved!")
                        else:
                            st.warning("Please enter some notes before saving.")

                with col2:
                    if st.button("📥 Download Notes"):
                        if 'saved_personal_notes' in st.session_state and st.session_state.saved_personal_notes:
                            # Combine all saved notes for text download
                            all_notes = "".join(st.session_state.saved_personal_notes)
                            st.download_button(
                                label="📄 Download Personal Notes (TXT)",
                                data=all_notes,
                                file_name=f"personal_notes_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain"
                            )

                            # Generate PDF for personal notes
                            try:
                                from pdf_generator import pdf_generator
                                pdf_bytes = pdf_generator.generate_notes_pdf(
                                    st.session_state.saved_personal_notes,
                                    "Personal Notes"
                                )
                                st.download_button(
                                    label="📕 Download Personal Notes (PDF)",
                                    data=pdf_bytes,
                                    file_name=f"personal_notes_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                    mime="application/pdf"
                                )
                            except Exception as e:
                                st.error(f"PDF generation failed: {str(e)}")
                        else:
                            st.warning("No saved notes to download. Save some notes first!")
            else:
                # No QuickNotes available - offer to generate them
                st.warning("⚠️ QuickNotes not available")
                st.info("QuickNotes may have failed to generate or are empty. Try regenerating them.")

                # Add button to manually regenerate QuickNotes
                if st.button("🔄 Generate QuickNotes", type="primary"):
                    if hasattr(st.session_state, 'processed_data') and st.session_state.processed_data:
                        with st.spinner("Generating QuickNotes..."):
                            try:
                                current_learning_mode = getattr(st.session_state, 'learning_mode', 'Intermediate')
                                if Config.GOOGLE_API_KEY:
                                    quick_notes = st.session_state.llm_generator._generate_quick_notes_with_gemini(
                                        st.session_state.processed_data['total_text'],
                                        current_learning_mode
                                    )
                                else:
                                    quick_notes = st.session_state.llm_generator.generate_quick_notes(
                                        st.session_state.processed_data['total_text'],
                                        current_learning_mode
                                    )

                                if quick_notes and quick_notes.strip():
                                    st.session_state.quick_notes = quick_notes
                                    setattr(st.session_state, f'quick_notes_{current_learning_mode}', quick_notes)
                                    st.success("✅ QuickNotes generated successfully!")
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to generate QuickNotes. Please try again.")
                            except Exception as e:
                                st.error(f"❌ Error generating QuickNotes: {str(e)}")
                    else:
                        st.error("❌ No document data available. Please upload a PDF first.")

        st.markdown("</div>", unsafe_allow_html=True)

    elif st.session_state.page == "Quiz":
        st.markdown("<div class='container-card'>", unsafe_allow_html=True)

        if not st.session_state.quiz_generated:
            st.title("🧠 Quiz")
            if not st.session_state.pdf_uploaded:
                st.info("Upload a PDF first to generate quiz questions!")
            else:
                st.info("Generate quiz questions from your uploaded PDF!")
                if st.button("🧠 Generate Quiz", type="primary"):
                    success = generate_quiz()
                    if success:
                        st.rerun()
        else:
            # Start quiz if not started
            if not st.session_state.quiz_started:
                st.title("🧠 Ready to Start Quiz?")
                quiz_summary = st.session_state.quiz_manager.get_quiz_summary()

                st.write(f"**Quiz:** {quiz_summary['quiz_title']}")
                st.write(f"**Questions:** {quiz_summary['total_questions']}")
                st.write(f"**Estimated Time:** {quiz_summary['total_questions'] * 2} minutes")

                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("🚀 Start Quiz", type="primary"):
                        st.session_state.quiz_manager.start_quiz()
                        st.session_state.quiz_started = True
                        st.rerun()

                with col2:
                    if st.button("🔄 Regenerate Quiz"):
                        success = generate_quiz()
                        if success:
                            st.rerun()

            # Quiz in progress
            elif st.session_state.quiz_started and not st.session_state.quiz_completed:
                current_question = st.session_state.quiz_manager.get_current_question()

                if current_question is None:
                    st.session_state.quiz_completed = True
                    st.session_state.page = "Result"
                    st.rerun()
                else:
                    # Progress bar
                    progress = st.session_state.quiz_manager.get_quiz_progress()
                    pct = int(progress['progress'])

                    st.markdown(f"<div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:8px'><div style='font-weight:700'>Your Progress</div><div style='color:#6b7280'>{pct}%</div></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='progress-track'><div class='progress-bar' style='width:{pct}%'></div></div>", unsafe_allow_html=True)
                    st.write("\n")

                    # Question display
                    st.markdown("<div class='question-card'>", unsafe_allow_html=True)
                    st.subheader(f"Question {current_question['question_number']} of {current_question['total_questions']}")
                    st.markdown(f"<h3 style='margin-top:6px'>{current_question['q']}</h3>", unsafe_allow_html=True)

                    # Answer choices
                    choice_idx = st.radio(
                        "Select your answer",
                        options=list(range(len(current_question['choices']))),
                        format_func=lambda i: current_question['choices'][i],
                        key=f"quiz_radio_{current_question['question_number']}",
                        label_visibility="collapsed"
                    )

                    # Initialize answer feedback state if not exists
                    if 'answer_submitted' not in st.session_state:
                        st.session_state.answer_submitted = False
                    if 'answer_feedback' not in st.session_state:
                        st.session_state.answer_feedback = None

                    # Submit answer button (only show if answer not yet submitted)
                    if not st.session_state.answer_submitted and st.button("Submit Answer →", type="primary"):
                        try:
                            if choice_idx is None:
                                st.warning("Please select an answer before submitting.")
                            else:
                                feedback = st.session_state.quiz_manager.submit_answer(choice_idx)

                                if not feedback:
                                    st.error("Failed to process your answer. Please try again.")
                                else:
                                    # Store feedback and mark as submitted
                                    st.session_state.answer_feedback = feedback
                                    st.session_state.answer_submitted = True
                                    st.rerun()
                        except Exception as submit_error:
                            st.error(f"Error submitting answer: {str(submit_error)}")
                            st.warning("Please try selecting your answer again.")

                    # Show feedback if answer has been submitted
                    if st.session_state.answer_submitted and st.session_state.answer_feedback:
                        feedback = st.session_state.answer_feedback

                        if feedback.get('is_correct'):
                            st.success("✅ Correct!")
                            # For correct answers, automatically move to next question after brief pause
                            try:
                                has_next = st.session_state.quiz_manager.next_question()
                                if not has_next:
                                    st.session_state.quiz_completed = True
                                    st.session_state.page = "Result"

                                # Reset answer state for next question
                                st.session_state.answer_submitted = False
                                st.session_state.answer_feedback = None

                                time.sleep(2)  # Brief pause to show feedback
                                st.rerun()
                            except Exception as next_error:
                                st.error(f"Error moving to next question: {str(next_error)}")
                        else:
                            st.error("❌ Incorrect")
                            explanation = feedback.get('explanation', 'No explanation available.')

                            # Debug: Show what we got in feedback
                            if not explanation or explanation == 'No explanation available.':
                                st.warning("⚠️ Explanation not available for this question.")
                                # Try to get explanation from the current question
                                current_q = st.session_state.quiz_manager.get_current_question()
                                if current_q and 'explanation' in current_q:
                                    explanation = current_q['explanation']
                                else:
                                    explanation = "Please refer to the source material for more details about this topic."

                            if explanation:
                                st.markdown(f"<div class='explanation'><strong>Explanation:</strong><div style='margin-top:8px'>{explanation}</div></div>", unsafe_allow_html=True)

                            # Show Next button for incorrect answers
                            if st.button("Next Question →", type="secondary", key="next_after_wrong"):
                                try:
                                    has_next = st.session_state.quiz_manager.next_question()
                                    if not has_next:
                                        st.session_state.quiz_completed = True
                                        st.session_state.page = "Result"

                                    # Reset answer state for next question
                                    st.session_state.answer_submitted = False
                                    st.session_state.answer_feedback = None
                                    st.rerun()
                                except Exception as next_error:
                                    st.error(f"Error moving to next question: {str(next_error)}")

                    st.markdown("</div>", unsafe_allow_html=True)

            # Quiz completed - redirect to results
            else:
                st.session_state.page = "Result"
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    elif st.session_state.page == "Result":
        st.markdown("<div class='container-card'>", unsafe_allow_html=True)

        if not st.session_state.quiz_completed:
            st.title("📊 Quiz Results")
            st.info("Complete a quiz to see your results!")

            if st.session_state.quiz_generated:
                if st.button("🚀 Start Quiz", type="primary"):
                    st.session_state.page = "Quiz"
                    st.rerun()
        else:
            # Calculate and display results
            score_report = st.session_state.quiz_manager.calculate_final_score()

            st.title("🎉 Quiz Results")

            # Main metrics
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                st.metric(
                    "Score",
                    f"{score_report['score_percentage']}%",
                    f"{score_report['correct_answers']}/{score_report['total_questions']}"
                )
            with col2:
                st.metric(
                    "Performance",
                    score_report['performance_level'],
                    f"{score_report['incorrect_answers']} incorrect"
                )
            with col3:
                st.metric(
                    "Time Taken",
                    score_report['time_taken_formatted'],
                    "Total duration"
                )

            # Performance visualization
            st.markdown("### 📈 Performance Overview")

            # Generate real topics from quiz questions
            real_topics = []
            if hasattr(st.session_state, 'quiz_data') and st.session_state.quiz_data:
                try:
                    real_topics = st.session_state.llm_generator.generate_topics_from_quiz(
                        st.session_state.quiz_data.get('questions', [])
                    )
                except Exception as e:
                    logger.error(f"Error generating topics: {e}")

            # Get topic performance data
            try:
                topic_performance = st.session_state.quiz_manager.get_topic_performance(real_topics)

                if topic_performance and len(topic_performance) > 0:
                    # Create DataFrame for the chart
                    perf_df = pd.DataFrame({
                        "Topic": list(topic_performance.keys()),
                        "Score": list(topic_performance.values())
                    })

                    # Create bar chart
                    fig = px.bar(
                        perf_df,
                        x="Topic",
                        y="Score",
                    title="Performance by Topic",
                    color="Score",
                    color_continuous_scale="RdYlGn",
                    height=400
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("📊 Performance data will be available after completing the quiz.")

            except Exception as e:
                st.error(f"Error displaying performance chart: {str(e)}")
                st.info("📊 Performance visualization temporarily unavailable.")

            # Review incorrect answers
            incorrect_answers = st.session_state.quiz_manager.get_incorrect_answers()

            if incorrect_answers:
                st.markdown("### ❌ Review Incorrect Answers")
                for answer in incorrect_answers:
                    with st.expander(f"Question {answer['question_number']}: {answer['question'][:60]}..."):
                        st.write(f"**Your answer:** {answer['your_answer']}")
                        st.write(f"**Correct answer:** {answer['correct_answer']}")
                        st.write(f"**Explanation:** {answer['explanation']}")

                        # Add option to ask follow-up questions
                        follow_up = st.text_input(
                            "Ask a follow-up question:",
                            key=f"followup_{answer['question_number']}",
                            placeholder="Need more clarification on this topic?"
                        )

                        if follow_up and st.button(f"Get Answer", key=f"followup_btn_{answer['question_number']}"):
                            if st.session_state.embeddings_ready:
                                with st.spinner("Finding relevant information..."):
                                    context = st.session_state.embeddings_manager.get_context_for_question(follow_up)
                                    answer_text = st.session_state.llm_generator.answer_question(follow_up, context)
                                    st.markdown("### 💡 Answer:")
                                    st.text_area("", value=answer_text, height=150, disabled=True, label_visibility="collapsed")
                            else:
                                st.warning("Embeddings not ready for Q&A.")
            else:
                st.success("🎉 Perfect score! No incorrect answers to review.")

            # Action buttons
            st.markdown("### 🔄 What's Next?")
            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                if st.button("🔄 Retake Quiz", type="secondary"):
                    restart_quiz()
                    st.session_state.page = "Quiz"
                    st.rerun()

            with col2:
                if st.button("📚 Review Material", type="secondary"):
                    st.session_state.page = "Normal"
                    st.rerun()

            with col3:
                if st.button("📝 Study Notes", type="secondary"):
                    st.session_state.page = "QuickNotes"
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

# footer spacing
st.write("\n")

# End of file
