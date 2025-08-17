"""
Local configuration overrides for StudyMate.ai
Copy and modify this file to customize settings
"""

from config import Config

class LocalConfig(Config):
    """Local configuration overrides"""
    
    # Uncomment and modify these settings as needed
    
    # PDF Processing
    # MAX_FILE_SIZE_MB = 25  # Reduce for limited memory
    # PDF_CHUNK_SIZE = 500   # Smaller chunks for faster processing
    
    # Model Settings
    # DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    # DEFAULT_LLM_MODEL = "gpt2"  # Lighter model
    
    # Quiz Settings
    # DEFAULT_QUIZ_QUESTIONS = 5  # Fewer questions for testing
    
    # Performance
    # EMBEDDING_BATCH_SIZE = 16  # Smaller batch for limited memory
    # ENABLE_CACHING = True
    
    # Debug
    # DEBUG = True
    # LOG_LEVEL = "DEBUG"

# To use this config, import it in your main application:
# from local_config import LocalConfig as Config
