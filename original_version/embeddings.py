"""
Embeddings and Vector Search Module for StudyMate.ai
Handles text embeddings using HuggingFace models and FAISS vector search
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import logging
import pickle
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingsManager:
    """Manages text embeddings and vector search using FAISS with IBM Granite models"""

    def __init__(self, model_name: str = "ibm-granite/granite-embedding-30m-english"):
        """
        Initialize embeddings manager with IBM Granite model

        Args:
            model_name: HuggingFace model name for embeddings (IBM Granite preferred)
        """
        self.model_name = model_name
        self.model = None
        self.index = None
        self.chunks_metadata = []
        self.embedding_dimension = None

        # Load the embedding model
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model with IBM Granite preference"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")

            # Try to load IBM Granite model first
            if "granite" in self.model_name.lower():
                try:
                    self.model = SentenceTransformer(self.model_name, trust_remote_code=True)
                    logger.info("Successfully loaded IBM Granite embedding model")
                except Exception as granite_error:
                    logger.warning(f"Failed to load IBM Granite model: {str(granite_error)}")
                    logger.info("Falling back to alternative embedding model")
                    self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
                    self.model = SentenceTransformer(self.model_name)
            else:
                self.model = SentenceTransformer(self.model_name)

            # Get embedding dimension
            test_embedding = self.model.encode(["test"])
            self.embedding_dimension = test_embedding.shape[1]

            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dimension}")

        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            # Fallback to a smaller model if the primary fails
            try:
                logger.info("Trying fallback model: all-MiniLM-L6-v2")
                self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
                self.model = SentenceTransformer(self.model_name)
                test_embedding = self.model.encode(["test"])
                self.embedding_dimension = test_embedding.shape[1]
                logger.info("Fallback model loaded successfully")
            except Exception as fallback_error:
                logger.error(f"Fallback model also failed: {str(fallback_error)}")
                raise Exception(f"Failed to load any embedding model: {str(fallback_error)}")
    
    def create_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """
        Create embeddings for text chunks
        
        Args:
            chunks: List of text chunks with metadata
            
        Returns:
            Numpy array of embeddings
        """
        try:
            if not chunks:
                raise ValueError("No chunks provided for embedding")
            
            # Extract text from chunks
            texts = [chunk['text'] for chunk in chunks]
            
            logger.info(f"Creating embeddings for {len(texts)} chunks")
            
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise Exception(f"Failed to create embeddings: {str(e)}")
    
    def build_faiss_index(self, embeddings: np.ndarray, chunks_metadata: List[Dict]):
        """
        Build FAISS index for vector search
        
        Args:
            embeddings: Numpy array of embeddings
            chunks_metadata: Metadata for each chunk
        """
        try:
            if embeddings.shape[0] == 0:
                raise ValueError("No embeddings provided")
            
            # Store metadata
            self.chunks_metadata = chunks_metadata
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add embeddings to index
            self.index.add(embeddings.astype('float32'))
            
            logger.info(f"Built FAISS index with {self.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Error building FAISS index: {str(e)}")
            raise Exception(f"Failed to build FAISS index: {str(e)}")
    
    def search_similar_chunks(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for similar chunks using semantic similarity
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of similar chunks with scores
        """
        try:
            if self.index is None:
                raise ValueError("FAISS index not built. Call build_faiss_index first.")
            
            # Create query embedding
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            
            # Normalize query embedding
            faiss.normalize_L2(query_embedding)
            
            # Search in FAISS index
            scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            # Prepare results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.chunks_metadata):
                    result = {
                        'rank': i + 1,
                        'score': float(score),
                        'chunk_data': self.chunks_metadata[idx],
                        'text': self.chunks_metadata[idx]['text']
                    }
                    results.append(result)
            
            logger.info(f"Found {len(results)} similar chunks for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            raise Exception(f"Semantic search failed: {str(e)}")
    
    def get_context_for_question(self, query: str, max_context_length: int = 2000) -> str:
        """
        Get relevant context for answering a question
        
        Args:
            query: User question
            max_context_length: Maximum length of context to return
            
        Returns:
            Concatenated context from relevant chunks
        """
        try:
            # Search for relevant chunks
            similar_chunks = self.search_similar_chunks(query, top_k=5)
            
            # Build context from top chunks
            context_parts = []
            current_length = 0
            
            for chunk_result in similar_chunks:
                chunk_text = chunk_result['text']
                page_num = chunk_result['chunk_data'].get('page_number', 'Unknown')
                
                # Add page reference
                chunk_with_ref = f"[Page {page_num}] {chunk_text}"
                
                if current_length + len(chunk_with_ref) <= max_context_length:
                    context_parts.append(chunk_with_ref)
                    current_length += len(chunk_with_ref)
                else:
                    # Add partial chunk if there's space
                    remaining_space = max_context_length - current_length
                    if remaining_space > 100:  # Only add if meaningful space left
                        partial_chunk = chunk_with_ref[:remaining_space] + "..."
                        context_parts.append(partial_chunk)
                    break
            
            context = "\n\n".join(context_parts)
            logger.info(f"Built context of {len(context)} characters from {len(context_parts)} chunks")
            
            return context
            
        except Exception as e:
            logger.error(f"Error building context: {str(e)}")
            return ""
    
    def save_index(self, filepath: str):
        """
        Save FAISS index and metadata to disk
        
        Args:
            filepath: Path to save the index
        """
        try:
            if self.index is None:
                raise ValueError("No index to save")
            
            # Save FAISS index
            faiss.write_index(self.index, f"{filepath}.faiss")
            
            # Save metadata
            with open(f"{filepath}_metadata.pkl", 'wb') as f:
                pickle.dump({
                    'chunks_metadata': self.chunks_metadata,
                    'model_name': self.model_name,
                    'embedding_dimension': self.embedding_dimension
                }, f)
            
            logger.info(f"Saved index and metadata to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            raise Exception(f"Failed to save index: {str(e)}")
    
    def load_index(self, filepath: str):
        """
        Load FAISS index and metadata from disk
        
        Args:
            filepath: Path to load the index from
        """
        try:
            # Load FAISS index
            if os.path.exists(f"{filepath}.faiss"):
                self.index = faiss.read_index(f"{filepath}.faiss")
            else:
                raise FileNotFoundError(f"Index file not found: {filepath}.faiss")
            
            # Load metadata
            if os.path.exists(f"{filepath}_metadata.pkl"):
                with open(f"{filepath}_metadata.pkl", 'rb') as f:
                    metadata = pickle.load(f)
                    self.chunks_metadata = metadata['chunks_metadata']
                    self.embedding_dimension = metadata['embedding_dimension']
            else:
                raise FileNotFoundError(f"Metadata file not found: {filepath}_metadata.pkl")
            
            logger.info(f"Loaded index with {self.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            raise Exception(f"Failed to load index: {str(e)}")
    
    def get_chunk_by_page(self, page_number: int) -> List[Dict]:
        """
        Get all chunks from a specific page
        
        Args:
            page_number: Page number to filter by
            
        Returns:
            List of chunks from the specified page
        """
        page_chunks = []
        for chunk in self.chunks_metadata:
            if chunk.get('page_number') == page_number:
                page_chunks.append(chunk)
        
        return page_chunks
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the embeddings and index
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_chunks': len(self.chunks_metadata),
            'embedding_dimension': self.embedding_dimension,
            'model_name': self.model_name,
            'index_size': self.index.ntotal if self.index else 0
        }
        
        if self.chunks_metadata:
            # Page statistics
            pages = set(chunk.get('page_number', 0) for chunk in self.chunks_metadata)
            stats['total_pages'] = len(pages)
            stats['chunks_per_page'] = len(self.chunks_metadata) / len(pages) if pages else 0
        
        return stats
