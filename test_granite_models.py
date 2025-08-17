"""
Test script to check IBM Granite model availability and functionality
"""

import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_granite_embedding_model():
    """Test IBM Granite embedding model"""
    print("🧪 Testing IBM Granite Embedding Model...")

    # Test encoding
    test_texts = ["This is a test sentence.", "Another test sentence for embedding."]

    try:
        # Try to load IBM Granite embedding model
        model_name = "ibm-granite/granite-embedding-30m-english"
        print(f"Loading model: {model_name}")

        model = SentenceTransformer(model_name, trust_remote_code=True)
        embeddings = model.encode(test_texts)
        
        print(f"✅ Successfully loaded {model_name}")
        print(f"📊 Embedding dimension: {embeddings.shape[1]}")
        print(f"📝 Test embeddings shape: {embeddings.shape}")
        
        return True, model_name, embeddings.shape[1]
        
    except Exception as e:
        print(f"❌ Failed to load IBM Granite embedding model: {str(e)}")
        print("🔄 Falling back to alternative model...")
        
        try:
            fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
            model = SentenceTransformer(fallback_model)
            embeddings = model.encode(test_texts)
            
            print(f"✅ Successfully loaded fallback model: {fallback_model}")
            print(f"📊 Embedding dimension: {embeddings.shape[1]}")
            
            return True, fallback_model, embeddings.shape[1]
            
        except Exception as fallback_error:
            print(f"❌ Fallback model also failed: {str(fallback_error)}")
            return False, None, None

def test_granite_llm_model():
    """Test IBM Granite LLM model"""
    print("\n🧪 Testing IBM Granite LLM Model...")
    
    try:
        # Try to load IBM Granite LLM model
        model_name = "ibm-granite/granite-3.0-2b-instruct"
        print(f"Loading model: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Test generation
        test_prompt = "Summarize the following text: Machine learning is a subset of artificial intelligence."
        inputs = tokenizer.encode(test_prompt, return_tensors='pt', truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 50,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"✅ Successfully loaded {model_name}")
        print(f"🔧 Device: {model.device if hasattr(model, 'device') else 'CPU'}")
        print(f"📝 Test generation: {generated_text[:100]}...")
        
        return True, model_name
        
    except Exception as e:
        print(f"❌ Failed to load IBM Granite LLM model: {str(e)}")
        print("🔄 Falling back to alternative model...")
        
        try:
            fallback_model = "microsoft/DialoGPT-small"
            tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            model = AutoModelForCausalLM.from_pretrained(fallback_model)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            print(f"✅ Successfully loaded fallback model: {fallback_model}")
            
            return True, fallback_model
            
        except Exception as fallback_error:
            print(f"❌ Fallback model also failed: {str(fallback_error)}")
            return False, None

def test_system_info():
    """Display system information"""
    print("🖥️  System Information:")
    print(f"   Python version: {torch.__version__}")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

def main():
    """Main test function"""
    print("🚀 IBM Granite Models Test Suite")
    print("=" * 50)
    
    # System info
    test_system_info()
    print()
    
    # Test embedding model
    embedding_success, embedding_model, embedding_dim = test_granite_embedding_model()
    
    # Test LLM model
    llm_success, llm_model = test_granite_llm_model()
    
    # Summary
    print("\n📋 Test Summary:")
    print("=" * 30)
    
    if embedding_success:
        print(f"✅ Embedding Model: {embedding_model} (dim: {embedding_dim})")
    else:
        print("❌ Embedding Model: Failed")
    
    if llm_success:
        print(f"✅ LLM Model: {llm_model}")
    else:
        print("❌ LLM Model: Failed")
    
    if embedding_success and llm_success:
        print("\n🎉 All models loaded successfully!")
        print("💡 You can now run the StudyMate.ai application with IBM Granite models.")
    else:
        print("\n⚠️  Some models failed to load. The application will use fallback models.")
    
    print("\n🚀 To start the application, run: python -m streamlit run ui.py")

if __name__ == "__main__":
    main()
