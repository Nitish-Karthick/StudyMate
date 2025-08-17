# StudyMate.ai - AI-Powered Study Companion

A comprehensive AI-powered study application that helps students learn from PDF documents through intelligent summaries, quizzes, and interactive Q&A.

## Features

- **PDF Processing**: Upload and process PDF documents with intelligent text extraction
- **AI-Powered Summaries**: Generate page-by-page and overall document summaries
- **QuickNotes**: Create concise study notes adapted to your learning level (Beginner/Intermediate/Advanced)
- **Interactive Quizzes**: Auto-generated multiple choice questions with explanations
- **Q&A System**: Ask questions about your documents and get AI-powered answers
- **Personal Notes**: Take and save personal notes with download options
- **Learning Modes**: Adaptive content based on your learning level
- **PDF Export**: Download your notes and summaries as PDF files

## Technology Stack

- **Frontend**: Streamlit
- **AI Models**: 
  - Google Gemini API (Enhanced features)
  - IBM Granite Embeddings (Document understanding)
- **PDF Processing**: PyMuPDF
- **Vector Search**: FAISS
- **PDF Generation**: ReportLab

## Quick Start

### Local Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd studymate-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run ui.py
```

## Usage

1. **Upload PDF**: Click "Choose a PDF file" and upload your study material
2. **Process Document**: Choose between Quick Process (progressive) or Full Process
3. **Generate Content**: 
   - View page summaries and overall document summary
   - Generate QuickNotes for your learning level
   - Create and take quizzes
   - Ask questions about the content
4. **Take Notes**: Use the personal notes feature to add your own insights
5. **Download**: Export your notes and summaries as TXT or PDF files

## Learning Modes

- **Beginner**: Simple language, basic concepts, step-by-step explanations
- **Intermediate**: Balanced complexity, moderate detail, practical examples
- **Advanced**: Technical depth, complex concepts, comprehensive analysis

