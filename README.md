# StudyMate.ai - AI-Powered Study Assistant

A comprehensive full-stack AI application that transforms PDF documents into interactive study materials with summaries, quizzes, and intelligent Q&A capabilities.

## 🚀 Features

- **PDF Processing**: Upload and extract text from PDF documents with page-by-page breakdown
- **AI Summaries**: Generate page-by-page summaries and condensed QuickNotes
- **Interactive Quizzes**: Auto-generated multiple choice questions with explanations
- **Semantic Search**: Ask questions about your documents with AI-powered answers
- **Progress Tracking**: Monitor quiz performance and review incorrect answers
- **Modern UI**: Clean, responsive interface with light/dark theme support

## 🛠️ Tech Stack

- **Frontend**: Streamlit for interactive web interface
- **Backend**: Python with modular architecture
- **PDF Processing**: PyMuPDF for text extraction
- **Embeddings**: HuggingFace Sentence Transformers
- **Vector Search**: FAISS for semantic similarity
- **LLM Generation**: Local models with IBM Watsonx/Mistral integration
- **Visualization**: Plotly for performance charts

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- At least 4GB RAM (8GB recommended)
- Internet connection for model downloads

## 🔧 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd studymate-ai
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
