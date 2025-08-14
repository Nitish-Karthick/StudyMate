"""
Configuration file for StudyMate.ai
Contains model settings, API keys, and application configuration
"""

import os
from typing import Dict, Any

class Config:
    """Configuration class for StudyMate.ai application"""
    
    # Application Settings
    APP_NAME = "StudyMate.ai"
    APP_VERSION = "1.0.0"
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # File Upload Settings
    MAX_FILE_SIZE_MB = 50
    ALLOWED_FILE_TYPES = ["pdf"]
    UPLOAD_TIMEOUT_SECONDS = 300
    
    # PDF Processing Settings
    PDF_CHUNK_SIZE = 1000
    PDF_CHUNK_OVERLAP = 200
    MAX_PAGES_TO_PROCESS = 100
    
    # Embedding Model Settings - IBM Granite preferred
    DEFAULT_EMBEDDING_MODEL = "ibm-granite/granite-embedding-30m-english"
    FALLBACK_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE = 32
    EMBEDDING_DIMENSION = 384  # Will be determined at runtime
    
    # FAISS Settings
    FAISS_INDEX_TYPE = "IndexFlatIP"  # Inner Product for cosine similarity
    FAISS_SEARCH_K = 5  # Number of similar chunks to retrieve
    MAX_CONTEXT_LENGTH = 2000  # Maximum context length for Q&A
    
    # LLM Generation Settings - IBM Granite 3.0 2B as primary model
    DEFAULT_LLM_MODEL = "ibm-granite/granite-3.0-2b-instruct"  # Primary 2B parameter model
    FALLBACK_LLM_MODEL = "microsoft/DialoGPT-small"  # Lightweight fallback
    GRANITE_LLM_MODEL = "ibm-granite/granite-3.0-2b-instruct"  # Same as default
    USE_LOCAL_MODELS = True  # Enable local model loading
    PREFER_API_GENERATION = True  # Prefer API but keep local as backup
    MAX_GENERATION_LENGTH = 200
    GENERATION_TEMPERATURE = 0.7
    GENERATION_TIMEOUT_SECONDS = 60

    # API Keys for Enhanced Features - Your DeepSeek configuration
    OPENROUTER_API_KEY = "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"
    OPENAI_API_KEY = None  # Not used - using custom endpoint instead
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    GOOGLE_API_KEY = "AIzaSyA9JEBW5u-xaOJz2gApcpwIwylZZkD09XI"  # Primary Gemini API key
    GOOGLE_API_KEY_BACKUP = "AIzaSyBl3trSPBEkHNWNqUTxaKI1NiZaHUevtbU"  # Backup Gemini API key

    # OpenRouter API Settings - Correct endpoint
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    OPENROUTER_MODEL = "deepseek/deepseek-r1:free"  # DeepSeek R1 Free model

    # Google Gemini API Settings - Fallback option
    GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    GEMINI_MODEL = "gemini-2.0-flash"  # Latest Gemini model

    # Alternative models (if you want to switch):
    # OPENROUTER_MODEL = "anthropic/claude-3-haiku"      # Fast, cheap
    # OPENROUTER_MODEL = "anthropic/claude-3-sonnet"     # Balanced
    # OPENROUTER_MODEL = "openai/gpt-4"                  # High quality
    # OPENROUTER_MODEL = "openai/gpt-4-turbo"            # Latest GPT-4
    # OPENROUTER_MODEL = "meta-llama/llama-3-8b-instruct" # Open source

    # API Model Settings
    OPENAI_MODEL = "gpt-3.5-turbo"
    ANTHROPIC_MODEL = "claude-3-haiku-20240307"
    GOOGLE_MODEL = "gemini-pro"
    
    # Quiz Settings
    DEFAULT_QUIZ_QUESTIONS = 10
    MIN_QUIZ_QUESTIONS = 5
    MAX_QUIZ_QUESTIONS = 20
    QUIZ_TIME_LIMIT_MINUTES = 30
    
    # Summary Settings
    MAX_SUMMARY_LENGTH = 300
    QUICKNOTES_MAX_LENGTH = 500
    
    # IBM Watsonx Settings (Optional - for production use)
    WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
    WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
    WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    WATSONX_MODEL = os.getenv("WATSONX_MODEL", "ibm/granite-13b-chat-v2")
    
    # Mistral Settings (Optional - for production use)
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
    MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-tiny")
    
    # Streamlit Settings
    STREAMLIT_THEME = "light"
    STREAMLIT_LAYOUT = "wide"
    STREAMLIT_SIDEBAR_STATE = "expanded"
    
    # Logging Settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Performance Settings
    ENABLE_CACHING = True
    CACHE_TTL_SECONDS = 3600  # 1 hour
    MAX_CONCURRENT_REQUESTS = 5
    
    # Error Handling Settings
    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 1
    ENABLE_FALLBACK_MODELS = True
    
    @classmethod
    def get_embedding_config(cls) -> Dict[str, Any]:
        """Get embedding model configuration"""
        return {
            "model_name": cls.DEFAULT_EMBEDDING_MODEL,
            "fallback_model": cls.FALLBACK_EMBEDDING_MODEL,
            "batch_size": cls.EMBEDDING_BATCH_SIZE,
            "dimension": cls.EMBEDDING_DIMENSION
        }
    
    @classmethod
    def get_llm_config(cls) -> Dict[str, Any]:
        """Get LLM configuration"""
        return {
            "model_name": cls.DEFAULT_LLM_MODEL,
            "fallback_model": cls.FALLBACK_LLM_MODEL,
            "max_length": cls.MAX_GENERATION_LENGTH,
            "temperature": cls.GENERATION_TEMPERATURE,
            "timeout": cls.GENERATION_TIMEOUT_SECONDS
        }
    
    @classmethod
    def get_pdf_config(cls) -> Dict[str, Any]:
        """Get PDF processing configuration"""
        return {
            "chunk_size": cls.PDF_CHUNK_SIZE,
            "chunk_overlap": cls.PDF_CHUNK_OVERLAP,
            "max_pages": cls.MAX_PAGES_TO_PROCESS,
            "max_file_size_mb": cls.MAX_FILE_SIZE_MB
        }
    
    @classmethod
    def get_quiz_config(cls) -> Dict[str, Any]:
        """Get quiz configuration"""
        return {
            "default_questions": cls.DEFAULT_QUIZ_QUESTIONS,
            "min_questions": cls.MIN_QUIZ_QUESTIONS,
            "max_questions": cls.MAX_QUIZ_QUESTIONS,
            "time_limit_minutes": cls.QUIZ_TIME_LIMIT_MINUTES
        }
    
    @classmethod
    def get_watsonx_config(cls) -> Dict[str, Any]:
        """Get IBM Watsonx configuration"""
        return {
            "api_key": cls.WATSONX_API_KEY,
            "project_id": cls.WATSONX_PROJECT_ID,
            "url": cls.WATSONX_URL,
            "model": cls.WATSONX_MODEL,
            "enabled": bool(cls.WATSONX_API_KEY and cls.WATSONX_PROJECT_ID)
        }
    
    @classmethod
    def get_mistral_config(cls) -> Dict[str, Any]:
        """Get Mistral configuration"""
        return {
            "api_key": cls.MISTRAL_API_KEY,
            "model": cls.MISTRAL_MODEL,
            "enabled": bool(cls.MISTRAL_API_KEY)
        }
    
    @classmethod
    def validate_config(cls) -> Dict[str, bool]:
        """Validate configuration settings"""
        validation_results = {
            "embedding_model_available": True,  # Will be checked at runtime
            "llm_model_available": True,  # Will be checked at runtime
            "file_size_valid": cls.MAX_FILE_SIZE_MB > 0 and cls.MAX_FILE_SIZE_MB <= 100,
            "chunk_size_valid": cls.PDF_CHUNK_SIZE > 0 and cls.PDF_CHUNK_SIZE <= 5000,
            "quiz_settings_valid": (cls.MIN_QUIZ_QUESTIONS <= cls.DEFAULT_QUIZ_QUESTIONS <= cls.MAX_QUIZ_QUESTIONS),
            "watsonx_configured": bool(cls.WATSONX_API_KEY and cls.WATSONX_PROJECT_ID),
            "mistral_configured": bool(cls.MISTRAL_API_KEY)
        }
        
        return validation_results
    
    @classmethod
    def get_environment_info(cls) -> Dict[str, Any]:
        """Get environment information"""
        return {
            "app_name": cls.APP_NAME,
            "app_version": cls.APP_VERSION,
            "debug_mode": cls.DEBUG,
            "python_version": os.sys.version,
            "environment_variables": {
                "WATSONX_API_KEY": "Set" if cls.WATSONX_API_KEY else "Not Set",
                "MISTRAL_API_KEY": "Set" if cls.MISTRAL_API_KEY else "Not Set",
                "DEBUG": os.getenv("DEBUG", "Not Set"),
                "LOG_LEVEL": cls.LOG_LEVEL
            }
        }

# Global configuration instance
config = Config()

# Environment-specific configurations
class DevelopmentConfig(Config):
    """Development environment configuration"""
    DEBUG = True
    LOG_LEVEL = "DEBUG"
    ENABLE_CACHING = False

class ProductionConfig(Config):
    """Production environment configuration"""
    DEBUG = False
    LOG_LEVEL = "WARNING"
    ENABLE_CACHING = True
    MAX_CONCURRENT_REQUESTS = 10

class TestingConfig(Config):
    """Testing environment configuration"""
    DEBUG = True
    LOG_LEVEL = "DEBUG"
    MAX_FILE_SIZE_MB = 10
    DEFAULT_QUIZ_QUESTIONS = 5

# Configuration factory
def get_config(environment: str = None) -> Config:
    """
    Get configuration based on environment
    
    Args:
        environment: Environment name (development, production, testing)
        
    Returns:
        Configuration instance
    """
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "development").lower()
    
    config_map = {
        "development": DevelopmentConfig,
        "production": ProductionConfig,
        "testing": TestingConfig
    }
    
    return config_map.get(environment, Config)()

# Export commonly used configurations
EMBEDDING_CONFIG = Config.get_embedding_config()
LLM_CONFIG = Config.get_llm_config()
PDF_CONFIG = Config.get_pdf_config()
QUIZ_CONFIG = Config.get_quiz_config()
WATSONX_CONFIG = Config.get_watsonx_config()
MISTRAL_CONFIG = Config.get_mistral_config()
