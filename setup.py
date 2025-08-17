"""
Setup script for StudyMate.ai
Helps with installation and environment setup
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_pip():
    """Check if pip is available"""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        print("✅ pip is available")
        return True
    except subprocess.CalledProcessError:
        print("❌ pip is not available")
        return False

def install_requirements():
    """Install required packages"""
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        print("📦 Installing requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def create_env_file():
    """Create .env file template"""
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    env_template = """# StudyMate.ai Environment Variables
# Optional: IBM Watsonx configuration
WATSONX_API_KEY=your_watsonx_api_key_here
WATSONX_PROJECT_ID=your_project_id_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com

# Optional: Mistral AI configuration
MISTRAL_API_KEY=your_mistral_api_key_here

# Application settings
DEBUG=False
LOG_LEVEL=INFO
ENVIRONMENT=development
"""
    
    try:
        with open(env_file, "w") as f:
            f.write(env_template)
        print("✅ Created .env template file")
        print("📝 Edit .env file to add your API keys (optional)")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def check_system_requirements():
    """Check system requirements"""
    print(f"🖥️  Operating System: {platform.system()} {platform.release()}")
    
    # Check available memory (rough estimate)
    try:
        if platform.system() == "Linux":
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemTotal" in line:
                        mem_kb = int(line.split()[1])
                        mem_gb = mem_kb / (1024 * 1024)
                        print(f"💾 Available RAM: {mem_gb:.1f} GB")
                        if mem_gb < 4:
                            print("⚠️  Warning: Less than 4GB RAM detected. Performance may be limited.")
                        break
    except:
        print("💾 Could not determine available RAM")
    
    return True

def test_imports():
    """Test if key packages can be imported"""
    test_packages = [
        ("streamlit", "Streamlit web framework"),
        ("fitz", "PyMuPDF for PDF processing"),
        ("sentence_transformers", "Sentence Transformers for embeddings"),
        ("faiss", "FAISS for vector search"),
        ("transformers", "HuggingFace Transformers"),
        ("pandas", "Pandas for data handling"),
        ("plotly", "Plotly for visualizations")
    ]
    
    print("\n🧪 Testing package imports...")
    all_good = True
    
    for package, description in test_packages:
        try:
            __import__(package)
            print(f"✅ {package}: {description}")
        except ImportError:
            print(f"❌ {package}: {description} - FAILED")
            all_good = False
    
    return all_good

def create_sample_config():
    """Create a sample configuration file"""
    config_file = Path("local_config.py")
    if config_file.exists():
        print("✅ local_config.py already exists")
        return True
    
    config_template = '''"""
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
'''
    
    try:
        with open(config_file, "w") as f:
            f.write(config_template)
        print("✅ Created local_config.py template")
        return True
    except Exception as e:
        print(f"❌ Failed to create local_config.py: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 StudyMate.ai Setup Script")
    print("=" * 40)
    
    # Check prerequisites
    if not check_python_version():
        return False
    
    if not check_pip():
        return False
    
    # System requirements
    check_system_requirements()
    
    # Install packages
    if not install_requirements():
        return False
    
    # Test imports
    if not test_imports():
        print("\n⚠️  Some packages failed to import. Try running:")
        print("pip install -r requirements.txt --upgrade")
        return False
    
    # Create configuration files
    create_env_file()
    create_sample_config()
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit .env file to add API keys (optional)")
    print("2. Run the application: streamlit run ui.py")
    print("3. Open http://localhost:8501 in your browser")
    print("4. Upload a PDF and start studying!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
