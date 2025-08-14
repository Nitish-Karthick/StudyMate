"""
PDF Processing Module for StudyMate.ai
Handles PDF text extraction, page-by-page processing, and text chunking
"""

import fitz  # PyMuPDF
import re
from typing import List, Dict, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handles PDF processing and text extraction"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize PDF processor

        Args:
            chunk_size: Maximum size of text chunks
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Test PyMuPDF installation
        try:
            # Create a simple test document to verify PyMuPDF works
            test_doc = fitz.open()
            test_doc.close()
            logger.info("PyMuPDF is working correctly")
        except Exception as test_error:
            logger.error(f"PyMuPDF test failed: {str(test_error)}")
            raise Exception(f"PyMuPDF is not working properly: {str(test_error)}")

    def test_pdf_file(self, pdf_bytes: bytes) -> bool:
        """
        Test if a PDF file can be opened and read

        Args:
            pdf_bytes: PDF file content as bytes

        Returns:
            True if PDF can be processed, False otherwise
        """
        test_doc = None
        try:
            test_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            page_count = test_doc.page_count
            logger.info(f"PDF test successful: {page_count} pages detected")
            return page_count > 0
        except Exception as e:
            logger.error(f"PDF test failed: {str(e)}")
            return False
        finally:
            if test_doc is not None:
                try:
                    test_doc.close()
                except:
                    pass

    def extract_text_from_pdf(self, pdf_file) -> Dict[str, any]:
        """
        Extract text from PDF file with page-by-page breakdown

        Args:
            pdf_file: Uploaded PDF file (Streamlit file object)

        Returns:
            Dictionary containing extracted text, page count, and metadata
        """
        doc = None
        try:
            # Read PDF from uploaded file
            pdf_bytes = pdf_file.read()

            if not pdf_bytes:
                raise Exception("PDF file is empty or could not be read")

            logger.info(f"Read {len(pdf_bytes)} bytes from PDF file")

            # Test PDF file first
            if not self.test_pdf_file(pdf_bytes):
                raise Exception("PDF file cannot be opened or is corrupted")

            # Open PDF document for processing
            try:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            except Exception as open_error:
                raise Exception(f"Failed to open PDF with PyMuPDF: {str(open_error)}")

            if doc.page_count == 0:
                raise Exception("PDF document has no pages")

            logger.info(f"Successfully opened PDF with {doc.page_count} pages")

            # Store page count immediately
            page_count = doc.page_count

            pages_text = []
            total_text = ""

            # Process all pages before closing document
            for page_num in range(page_count):
                try:
                    page = doc.load_page(page_num)
                    page_text = page.get_text()

                    # Clean the text
                    cleaned_text = self._clean_text(page_text)

                    pages_text.append({
                        'page_number': page_num + 1,
                        'text': cleaned_text,
                        'word_count': len(cleaned_text.split())
                    })

                    total_text += cleaned_text + "\n\n"

                except Exception as page_error:
                    logger.warning(f"Error processing page {page_num + 1}: {str(page_error)}")
                    # Continue with other pages
                    continue

            # Build result before closing document
            result = {
                'total_text': total_text.strip(),
                'pages': pages_text,
                'page_count': page_count,
                'total_word_count': len(total_text.split()),
                'filename': getattr(pdf_file, 'name', 'unknown.pdf')
            }

            logger.info(f"Successfully processed PDF: {result['filename']} with {page_count} pages")
            return result

        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise Exception(f"Failed to process PDF: {str(e)}")
        finally:
            # Always close document in finally block
            if doc is not None:
                try:
                    doc.close()
                    logger.debug("PDF document closed successfully")
                except Exception as close_error:
                    logger.warning(f"Error closing document: {str(close_error)}")
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing extra whitespace and formatting issues
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers (basic patterns)
        text = re.sub(r'\n\d+\n', '\n', text)
        
        # Remove special characters that might interfere with processing
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\'\/]', '', text)
        
        return text.strip()
    
    def create_text_chunks(self, text: str, page_number: int = None) -> List[Dict[str, any]]:
        """
        Split text into chunks for embedding
        
        Args:
            text: Text to chunk
            page_number: Optional page number for metadata
            
        Returns:
            List of text chunks with metadata
        """
        chunks = []
        
        # Split by sentences first
        sentences = re.split(r'[.!?]+', text)
        
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    'text': current_chunk.strip(),
                    'chunk_index': chunk_index,
                    'page_number': page_number,
                    'word_count': len(current_chunk.split())
                })
                
                # Start new chunk with overlap
                overlap_words = current_chunk.split()[-self.chunk_overlap//10:]  # Rough word-based overlap
                current_chunk = " ".join(overlap_words) + " " + sentence
                chunk_index += 1
            else:
                current_chunk += " " + sentence
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'chunk_index': chunk_index,
                'page_number': page_number,
                'word_count': len(current_chunk.split())
            })
        
        return chunks
    
    def process_pdf_for_embeddings(self, pdf_file) -> Dict[str, any]:
        """
        Complete PDF processing pipeline for embedding generation

        Args:
            pdf_file: Uploaded PDF file

        Returns:
            Processed data ready for embedding
        """
        try:
            logger.info(f"Starting PDF processing for embeddings: {getattr(pdf_file, 'name', 'unknown')}")

            # Extract text from PDF
            pdf_data = self.extract_text_from_pdf(pdf_file)

            if not pdf_data or not pdf_data.get('pages'):
                raise Exception("No text data extracted from PDF")
            
            # Create chunks for each page
            all_chunks = []
            page_summaries = []

            logger.info(f"Processing {len(pdf_data['pages'])} pages for chunking")

            for page_data in pdf_data['pages']:
                try:
                    page_chunks = self.create_text_chunks(
                        page_data['text'],
                        page_data['page_number']
                    )
                    all_chunks.extend(page_chunks)

                    # Prepare page summary data
                    page_summaries.append({
                        'page_number': page_data['page_number'],
                        'text': page_data['text'],
                        'word_count': page_data['word_count'],
                        'chunk_count': len(page_chunks)
                    })

                    logger.debug(f"Page {page_data['page_number']}: {len(page_chunks)} chunks created")

                except Exception as chunk_error:
                    logger.warning(f"Error creating chunks for page {page_data['page_number']}: {str(chunk_error)}")
                    continue
            
            # Validate that we have chunks
            if not all_chunks:
                raise Exception("No text chunks could be created from the PDF. The document might be empty or contain only images.")

            result = {
                'pdf_metadata': {
                    'filename': pdf_data['filename'],
                    'page_count': pdf_data['page_count'],
                    'total_word_count': pdf_data['total_word_count']
                },
                'pages': page_summaries,
                'chunks': all_chunks,
                'total_text': pdf_data['total_text']
            }

            logger.info(f"Successfully created {len(all_chunks)} chunks from {pdf_data['page_count']} pages")
            return result
            
        except Exception as e:
            logger.error(f"Error in PDF processing pipeline: {str(e)}")
            raise Exception(f"PDF processing failed: {str(e)}")
    
    def get_page_text(self, processed_data: Dict, page_number: int) -> str:
        """
        Get text for a specific page
        
        Args:
            processed_data: Output from process_pdf_for_embeddings
            page_number: Page number to retrieve
            
        Returns:
            Text content of the specified page
        """
        for page in processed_data['pages']:
            if page['page_number'] == page_number:
                return page['text']
        return ""
    
    def search_text_in_pages(self, processed_data: Dict, query: str) -> List[Dict]:
        """
        Simple text search across pages
        
        Args:
            processed_data: Output from process_pdf_for_embeddings
            query: Search query
            
        Returns:
            List of pages containing the query
        """
        results = []
        query_lower = query.lower()
        
        for page in processed_data['pages']:
            if query_lower in page['text'].lower():
                # Find the context around the match
                text_lower = page['text'].lower()
                match_index = text_lower.find(query_lower)
                
                # Extract context (100 characters before and after)
                start = max(0, match_index - 100)
                end = min(len(page['text']), match_index + len(query) + 100)
                context = page['text'][start:end]
                
                results.append({
                    'page_number': page['page_number'],
                    'context': context,
                    'full_text': page['text']
                })
        
        return results
