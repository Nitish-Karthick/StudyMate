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
  - IBM Granite 3.0 2B (Local inference)
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

### Online Deployment

#### Streamlit Cloud
1. Fork this repository
2. Connect to Streamlit Cloud
3. Deploy directly from GitHub

#### Heroku
1. Create a new Heroku app
2. Connect to your GitHub repository
3. Deploy with automatic builds

#### Railway/Render
1. Connect your GitHub repository
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `streamlit run ui.py --server.port $PORT --server.address 0.0.0.0`

## Configuration

The application uses API keys for enhanced features. You can:

1. **Use Local Models Only**: The app works without API keys using local IBM Granite models
2. **Add API Keys**: For enhanced features, configure in `config.py`:
   - Google Gemini API key for better content generation
   - OpenRouter API key for additional model access

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

## File Structure

```
studymate-ai/
├── ui.py                 # Main Streamlit application
├── config.py            # Configuration and API keys
├── llm_generator.py     # AI content generation
├── pdf_processor.py     # PDF text extraction
├── embeddings.py        # Document embeddings and search
├── quiz_manager.py      # Quiz generation and management
├── pdf_generator.py     # PDF export functionality
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For issues and questions, please create an issue in the GitHub repository.
