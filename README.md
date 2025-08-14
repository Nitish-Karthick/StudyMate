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

4. **Set up environment variables** (optional for advanced features):
   ```bash
   # Create .env file
   echo "WATSONX_API_KEY=your_watsonx_api_key" >> .env
   echo "WATSONX_PROJECT_ID=your_project_id" >> .env
   echo "MISTRAL_API_KEY=your_mistral_api_key" >> .env
   echo "DEBUG=False" >> .env
   ```

## 🚀 Quick Start

1. **Run the application**:
   ```bash
   streamlit run ui.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Upload a PDF** using the sidebar file uploader

4. **Wait for processing** - the app will:
   - Extract text from your PDF
   - Generate embeddings for semantic search
   - Create page summaries
   - Generate QuickNotes

5. **Explore features**:
   - **Normal Tab**: View page summaries and ask questions
   - **QuickNotes Tab**: See condensed study notes
   - **Quiz Tab**: Generate and take quizzes
   - **Result Tab**: Review performance and incorrect answers

## 📁 Project Structure

```
studymate-ai/
├── ui.py                 # Main Streamlit application
├── pdf_processor.py      # PDF text extraction and chunking
├── embeddings.py         # Text embeddings and FAISS search
├── llm_generator.py      # LLM-based text generation
├── quiz_manager.py       # Quiz creation and management
├── config.py            # Configuration settings
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## ⚙️ Configuration

The application can be configured through `config.py`:

- **File Upload**: Maximum file size, allowed types
- **PDF Processing**: Chunk size, overlap, page limits
- **Models**: Embedding and LLM model selection
- **Quiz Settings**: Number of questions, time limits
- **Performance**: Caching, concurrent requests

## 🔧 Advanced Setup

### IBM Watsonx Integration

1. Sign up for IBM Watsonx account
2. Get API key and project ID
3. Set environment variables:
   ```bash
   export WATSONX_API_KEY="your_api_key"
   export WATSONX_PROJECT_ID="your_project_id"
   ```

### Mistral AI Integration

1. Get Mistral API key from https://mistral.ai/
2. Set environment variable:
   ```bash
   export MISTRAL_API_KEY="your_mistral_key"
   ```

### Custom Models

Modify `config.py` to use different models:

```python
# Change embedding model
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Change LLM model
DEFAULT_LLM_MODEL = "microsoft/DialoGPT-medium"
```

## 🐛 Troubleshooting

### Common Issues

1. **Model Download Errors**:
   - Ensure stable internet connection
   - Check available disk space (models can be 1-2GB)
   - Try fallback models in config

2. **PDF Processing Fails**:
   - Ensure PDF contains readable text (not just images)
   - Check file size (default limit: 50MB)
   - Try a different PDF file

3. **Memory Issues**:
   - Reduce batch size in config
   - Use smaller embedding models
   - Process shorter documents

4. **Slow Performance**:
   - Enable GPU if available
   - Reduce number of quiz questions
   - Use smaller models

### Error Messages

- **"No text chunks found"**: PDF might be image-based or corrupted
- **"Embedding generation failed"**: Model loading issue, check internet
- **"Quiz generation failed"**: Content too short or model unavailable

## 📊 Performance Tips

1. **For better accuracy**: Use larger embedding models
2. **For faster processing**: Use smaller models and reduce batch sizes
3. **For memory efficiency**: Process documents in smaller chunks
4. **For production**: Enable caching and use GPU acceleration

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Issues**: Report bugs on GitHub Issues
- **Documentation**: Check inline code comments
- **Configuration**: Review `config.py` for all settings

## 🔮 Future Enhancements

- [ ] Multi-language support
- [ ] Document comparison features
- [ ] Collaborative study sessions
- [ ] Mobile app version
- [ ] Advanced analytics dashboard
- [ ] Integration with learning management systems

## 📈 Version History

- **v1.0.0**: Initial release with core features
  - PDF processing and text extraction
  - AI-generated summaries and quizzes
  - Semantic search and Q&A
  - Interactive web interface

---

**Built with ❤️ for students and educators worldwide**
