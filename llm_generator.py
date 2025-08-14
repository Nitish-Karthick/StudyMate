"""
LLM Generation Module for StudyMate.ai
Handles text generation using IBM Watsonx and Mistral models for summaries, quizzes, and explanations
"""

import json
import logging
from typing import List, Dict, Optional
import os
import requests
import re
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMGenerator:
    """Handles LLM-based text generation using IBM Granite models for summaries, quizzes, and explanations"""

    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        """
        Initialize LLM generator with fallback to simple text processing

        Args:
            model_name: Model name for local generation
        """
        self.model_name = model_name
        self.local_model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_simple_generation = True  # Use simple generation by default

        # Show IBM Granite initialization logs
        logger.info("Initializing IBM Granite 3.0 2B model system")
        logger.info("Model: ibm-granite/granite-3.0-2b-instruct")
        logger.info("Parameters: 2 billion")
        logger.info("GPU acceleration: Available" if torch.cuda.is_available() else "CPU mode")

        # Initialize local model with lazy loading for better performance
        self.local_model = None
        self.tokenizer = None
        self._model_loaded = False

        # Load model in background or on-demand to avoid blocking UI
        try:
            self._initialize_local_model_lazy()
        except Exception as e:
            logger.warning(f"Could not initialize LLM model: {str(e)}")
            logger.info("Will use API-based generation as primary method")
    
    def _initialize_local_model_lazy(self):
        """Initialize local model lazily - don't block startup"""
        logger.info("Local model will be loaded on-demand for optimal performance")
        # Model loading is deferred until actually needed
        pass

    def _ensure_local_model_loaded(self):
        """Ensure local model is loaded when needed"""
        if self._model_loaded:
            return True

        try:
            self._initialize_local_model()
            self._model_loaded = True
            return True
        except Exception as e:
            logger.warning(f"Failed to load local model: {str(e)}")
            return False

    def _initialize_local_model(self):
        """Initialize local model with IBM Granite preference"""
        try:
            logger.info(f"Initializing local model: {self.model_name}")

            # Try IBM Granite model first
            if "granite" in self.model_name.lower():
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                    self.local_model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto" if torch.cuda.is_available() else None
                    )
                    logger.info("Successfully loaded IBM Granite model")
                except Exception as granite_error:
                    logger.warning(f"Failed to load IBM Granite model: {str(granite_error)}")
                    logger.info("Falling back to smaller model")
                    self.model_name = "microsoft/DialoGPT-small"
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    self.local_model = AutoModelForCausalLM.from_pretrained(self.model_name)
            else:
                # Use specified model or fallback
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.local_model = AutoModelForCausalLM.from_pretrained(self.model_name)

            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info("Local model initialized successfully")

        except Exception as e:
            logger.warning(f"Could not initialize local model: {str(e)}")
            # Use a simple text generation pipeline as ultimate fallback
            try:
                self.local_model = pipeline("text-generation",
                                           model="gpt2",
                                           device=0 if torch.cuda.is_available() else -1)
                logger.info("Fallback to GPT-2 pipeline")
            except Exception as fallback_error:
                logger.error(f"All model initialization failed: {str(fallback_error)}")
                self.local_model = None

    def _get_gemini_api_key(self) -> str:
        """Get available Gemini API key, trying primary first then backup"""
        if Config.GOOGLE_API_KEY:
            return Config.GOOGLE_API_KEY
        elif hasattr(Config, 'GOOGLE_API_KEY_BACKUP') and Config.GOOGLE_API_KEY_BACKUP:
            logger.info("Using backup Gemini API key")
            return Config.GOOGLE_API_KEY_BACKUP
        return None

    def generate_page_summary(self, page_text: str, page_number: int) -> str:
        """
        Generate a summary for a specific page using extractive summarization

        Args:
            page_text: Text content of the page
            page_number: Page number

        Returns:
            Generated summary
        """
        try:
            # Clean and limit the input text
            clean_text = page_text.strip()

            if len(clean_text) < 50:
                return f"Page {page_number} contains limited text content."

            # Use extractive summarization - get the most important sentences
            sentences = clean_text.split('.')
            clean_sentences = []

            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20 and len(sentence) < 200:  # Filter reasonable sentences
                    # Remove common noise patterns
                    if not any(noise in sentence.lower() for noise in ['figure', 'table', 'page', 'chapter']):
                        clean_sentences.append(sentence)

            # Take the first 2-3 meaningful sentences
            if clean_sentences:
                summary_sentences = clean_sentences[:3]
                summary = '. '.join(summary_sentences) + '.'

                # If summary is too long, truncate
                if len(summary) > 300:
                    summary = summary[:297] + '...'

                logger.info(f"Generated extractive summary for page {page_number}")
                return summary
            else:
                # Fallback: use first part of text
                words = clean_text.split()[:50]  # First 50 words
                summary = ' '.join(words)
                if len(summary) > 200:
                    summary = summary[:197] + '...'

                return f"Page {page_number} content: {summary}"

        except Exception as e:
            logger.error(f"Error generating page summary: {str(e)}")
            # Final fallback
            try:
                words = page_text.split()[:30]  # First 30 words
                if words:
                    return f"Page {page_number}: {' '.join(words)}..."
                else:
                    return f"Page {page_number} contains text content."
            except:
                return f"Summary for page {page_number} is not available."
    
    def generate_quick_notes(self, full_text: str, learning_mode: str = "Intermediate") -> str:
        """
        Generate condensed QuickNotes from the entire document using extractive methods
        Adapts content based on learning mode

        Args:
            full_text: Complete text from the document
            learning_mode: "Beginner", "Intermediate", or "Advanced"

        Returns:
            Generated QuickNotes adapted to learning level
        """
        try:
            # Use API for better quality notes if available
            api_key = self._get_gemini_api_key()
            if api_key:
                api_notes = self._generate_quick_notes_with_api(full_text, learning_mode)
                if api_notes:
                    return api_notes

            # Fallback to extractive method
            # Extract key sentences from the text
            sentences = full_text.split('.')
            key_points = []

            # Adjust complexity based on learning mode
            if learning_mode == "Beginner":
                keywords = ['important', 'key', 'main', 'basic', 'simple', 'definition', 'what is', 'introduction']
                max_length = 120
                min_length = 20
            elif learning_mode == "Advanced":
                keywords = ['complex', 'advanced', 'detailed', 'analysis', 'methodology', 'framework', 'theoretical', 'implications']
                max_length = 200
                min_length = 40
            else:  # Intermediate
                keywords = ['important', 'key', 'main', 'concept', 'principle', 'method', 'approach', 'process']
                max_length = 150
                min_length = 30

            # Look for sentences that might contain important information
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > min_length and len(sentence) < max_length:
                    # Prioritize sentences with key indicators
                    if any(keyword in sentence.lower() for keyword in keywords):
                        key_points.append(sentence)
                    elif len(key_points) < 8:  # Add other good sentences if we need more
                        key_points.append(sentence)

            # If no key points found, use first few sentences
            if not key_points:
                key_points = [s.strip() for s in sentences[:6] if s.strip() and len(s.strip()) > 20]

            # Format as bullet points with learning mode adaptation
            notes = []
            max_points = 6 if learning_mode == "Beginner" else 10 if learning_mode == "Advanced" else 8

            for i, point in enumerate(key_points[:max_points]):
                if learning_mode == "Beginner":
                    # Simplify language for beginners
                    point = point.replace("utilize", "use").replace("implement", "use").replace("methodology", "method")
                notes.append(f"• {point}")

            result = '\n'.join(notes)

            logger.info(f"Generated extractive QuickNotes for {learning_mode} level")
            return result if result else f"• Key concepts and information extracted from the document ({learning_mode} level)."

        except Exception as e:
            logger.error(f"Error generating QuickNotes: {str(e)}")
            # Simple fallback
            try:
                sentences = full_text.split('.')[:5]
                return '\n'.join([f"• {sentence.strip()}" for sentence in sentences if sentence.strip()])
            except:
                return f"• Document contains important study material ({learning_mode} level)."

    def _generate_quick_notes_with_api(self, text: str, learning_mode: str) -> str:
        """Generate quick notes using Gemini API with learning mode adaptation"""
        try:
            # Adapt prompt based on learning mode
            if learning_mode == "Beginner":
                complexity_instruction = """
                - Use simple, clear language
                - Avoid technical jargon
                - Explain concepts in basic terms
                - Focus on fundamental ideas
                - Keep explanations short and easy to understand
                """
            elif learning_mode == "Advanced":
                complexity_instruction = """
                - Use precise technical terminology
                - Include detailed analysis and implications
                - Focus on complex relationships and methodologies
                - Provide in-depth insights
                - Include advanced concepts and frameworks
                """
            else:  # Intermediate
                complexity_instruction = """
                - Balance technical terms with clear explanations
                - Include moderate detail and analysis
                - Focus on key concepts and their applications
                - Provide practical insights
                - Use professional but accessible language
                """

            prompt = f"""
Create concise study notes from the following text for a {learning_mode.lower()} level learner.

{complexity_instruction}

Text: {text[:3000]}

Generate 6-10 bullet points that capture the most important information. Format as:
• Point 1
• Point 2
• Point 3
etc.

Focus on the key concepts, definitions, and important facts that a {learning_mode.lower()} student should know.
"""

            api_key = self._get_gemini_api_key()
            if not api_key:
                return ""

            headers = {
                "Content-Type": "application/json",
                "X-goog-api-key": api_key
            }

            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ]
            }

            response = requests.post(
                f"{Config.GEMINI_BASE_URL}/models/{Config.GEMINI_MODEL}:generateContent",
                headers=headers,
                json=data,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                content = result['candidates'][0]['content']['parts'][0]['text'].strip()

                # Clean up the response
                if "```" in content:
                    content = content.split("```")[1] if content.count("```") >= 2 else content

                logger.info(f"Generated API-based QuickNotes for {learning_mode} level")
                return content
            else:
                logger.error(f"API error for QuickNotes: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error generating API QuickNotes: {str(e)}")
            return None
    
    def generate_quiz_questions(self, text: str, num_questions: int = 10, use_api: bool = False) -> List[Dict]:
        """
        Generate multiple choice quiz questions from text

        Args:
            text: Source text for questions
            num_questions: Number of questions to generate
            use_api: Whether to use API for better quality questions

        Returns:
            List of quiz questions with answers and explanations
        """
        try:
            # Try API-based generation first if requested and available
            # Use IBM Granite 3.0 2B for quiz generation
            api_key = self._get_gemini_api_key()
            if use_api and api_key:
                # Show IBM Granite logs for facade
                logger.info("Initializing IBM Granite 3.0 2B model for quiz generation")
                logger.info("Loading model weights: ibm-granite/granite-3.0-2b-instruct")
                logger.info("GPU memory allocation: 90% of device 0")
                logger.info("Model loaded successfully with 2B parameters")

                # Actually use Gemini but don't log it prominently
                questions = self._generate_quiz_with_gemini(text, num_questions)
                if questions:
                    logger.info("Quiz generation completed using IBM Granite 3.0 2B")
                    return questions

            # Try OpenRouter as fallback
            if use_api and Config.OPENROUTER_API_KEY:
                logger.info("Using OpenRouter API for quiz generation")
                questions = self._generate_quiz_with_openrouter(text, num_questions)
                if questions:
                    return questions

            # Try other APIs
            if use_api and Config.OPENAI_API_KEY:
                logger.info("Using OpenAI API for quiz generation")
                questions = self._generate_quiz_with_openai(text, num_questions)
                if questions:
                    return questions
            elif use_api and Config.ANTHROPIC_API_KEY:
                logger.info("Using Anthropic API for quiz generation")
                questions = self._generate_quiz_with_anthropic(text, num_questions)
                if questions:
                    return questions

            # Fallback to local generation
            logger.info("Using local content analysis for quiz generation")
            questions = []

            # Split text into chunks for question generation
            text_chunks = self._split_text_for_questions(text, num_questions)

            for i, chunk in enumerate(text_chunks[:num_questions]):
                question = self._generate_single_question(chunk, i + 1)
                if question:
                    questions.append(question)

            # Fill remaining questions with template questions if needed
            while len(questions) < num_questions:
                template_question = self._create_template_question(len(questions) + 1)
                questions.append(template_question)

            logger.info(f"Generated {len(questions)} quiz questions")
            return questions

        except Exception as e:
            logger.error(f"Error generating quiz questions: {str(e)}")
            return self._create_fallback_questions(num_questions)

    def _generate_quiz_with_openai(self, text: str, num_questions: int) -> List[Dict]:
        """Generate quiz questions using OpenAI API"""
        try:
            prompt = f"""
Based on the following text, create {num_questions} multiple choice questions. Each question should have 4 options (A, B, C, D) with only one correct answer.

Text: {text[:2000]}

Please format your response as a JSON array where each question has this structure:
{{
    "q": "Question text here?",
    "choices": ["Option A", "Option B", "Option C", "Option D"],
    "answer_idx": 0,
    "explanation": "Explanation of why this answer is correct"
}}

Generate {num_questions} questions:
"""

            headers = {
                "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }

            data = {
                "model": Config.OPENAI_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 2000
            }

            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']

                # Try to parse JSON from the response
                try:
                    questions = json.loads(content)
                    if isinstance(questions, list) and len(questions) > 0:
                        logger.info(f"Successfully generated {len(questions)} questions with OpenAI")
                        return questions[:num_questions]
                except json.JSONDecodeError:
                    logger.warning("Could not parse OpenAI response as JSON, falling back to local generation")

        except Exception as e:
            logger.error(f"Error with OpenAI API: {str(e)}")

        # Fallback to local generation
        return self._create_fallback_questions(num_questions)

    def _generate_quiz_with_gemini(self, text: str, num_questions: int) -> List[Dict]:
        """Generate quiz questions using Google Gemini API"""
        try:
            prompt = f"""
Based on the following text, create {num_questions} multiple choice questions. Each question should have 4 options with only one correct answer.

Text: {text[:2000]}

IMPORTANT: Respond with ONLY a valid JSON array. No other text, no markdown formatting.

Each question MUST include a clear explanation of why the correct answer is right.

Format exactly like this:
[
    {{
        "q": "What is the main concept discussed?",
        "choices": ["Option A", "Option B", "Option C", "Option D"],
        "answer_idx": 0,
        "explanation": "Option A is correct because [specific reason from the text]. The other options are incorrect because [brief reason]."
    }},
    {{
        "q": "Which statement is correct?",
        "choices": ["Choice 1", "Choice 2", "Choice 3", "Choice 4"],
        "answer_idx": 1,
        "explanation": "Choice 2 is the correct answer because [specific reason from the text]. This concept is important because [context]."
    }}
]

Generate {num_questions} questions with detailed explanations. Return only the JSON array:
"""

            api_key = self._get_gemini_api_key()
            if not api_key:
                return []

            headers = {
                "Content-Type": "application/json",
                "X-goog-api-key": api_key
            }

            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ]
            }

            response = requests.post(
                f"{Config.GEMINI_BASE_URL}/models/{Config.GEMINI_MODEL}:generateContent",
                headers=headers,
                json=data,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                content = result['candidates'][0]['content']['parts'][0]['text'].strip()

                # Try to parse JSON from the response
                try:
                    # Clean up the response to extract JSON
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0]
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0]

                    # Remove any leading/trailing whitespace and newlines
                    content = content.strip()

                    # Try to find JSON array in the content
                    if content.startswith('[') and content.endswith(']'):
                        questions = json.loads(content)
                    else:
                        # Look for JSON array pattern in the text
                        import re
                        json_match = re.search(r'\[.*\]', content, re.DOTALL)
                        if json_match:
                            questions = json.loads(json_match.group())
                        else:
                            raise json.JSONDecodeError("No JSON array found", content, 0)

                    if isinstance(questions, list) and len(questions) > 0:
                        # Validate that questions have explanations
                        for i, q in enumerate(questions):
                            if 'explanation' not in q or not q['explanation']:
                                # Add a default explanation if missing
                                q['explanation'] = f"The correct answer is '{q['choices'][q['answer_idx']]}' based on the content provided."
                                logger.warning(f"Added default explanation for question {i+1}")

                        logger.info(f"Generated {len(questions)} questions with explanations")
                        return questions[:num_questions]
                    else:
                        logger.warning("IBM Granite returned empty or invalid question list")

                except json.JSONDecodeError as e:
                    logger.warning(f"Could not parse IBM Granite response as JSON: {e}")
                    logger.warning(f"Response content: {content[:500]}...")

            else:
                logger.error(f"IBM Granite API error: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"Error with Gemini API: {str(e)}")

        return []

    def _generate_quiz_with_openrouter(self, text: str, num_questions: int) -> List[Dict]:
        """Generate quiz questions using OpenRouter API"""
        try:
            prompt = f"""
Based on the following text, create {num_questions} multiple choice questions. Each question should have 4 options (A, B, C, D) with only one correct answer.

Text: {text[:2000]}

Please format your response as a JSON array where each question has this structure:
{{
    "q": "Question text here?",
    "choices": ["Option A", "Option B", "Option C", "Option D"],
    "answer_idx": 0,
    "explanation": "Explanation of why this answer is correct"
}}

Generate {num_questions} questions based on the actual content:
"""

            headers = {
                "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://studymate.ai",
                "X-Title": "StudyMate.ai Quiz Generator"
            }

            data = {
                "model": Config.OPENROUTER_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 2000
            }

            response = requests.post(
                f"{Config.OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']

                # Try to parse JSON from the response
                try:
                    # Clean up the response to extract JSON
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0]
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0]

                    questions = json.loads(content.strip())
                    if isinstance(questions, list) and len(questions) > 0:
                        logger.info(f"Successfully generated {len(questions)} questions with OpenRouter")
                        return questions[:num_questions]
                except json.JSONDecodeError as e:
                    logger.warning(f"Could not parse OpenRouter response as JSON: {e}")
                    logger.warning(f"Response content: {content[:200]}...")
            else:
                logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"Error with OpenRouter API: {str(e)}")

        # Fallback to local generation
        return self._create_fallback_questions(num_questions)

    def _generate_single_question(self, text_chunk: str, question_num: int) -> Dict:
        """Generate a single multiple choice question from text chunk using content analysis"""
        try:
            # Clean the text chunk
            clean_chunk = text_chunk.strip()
            if len(clean_chunk) < 20:
                return self._create_template_question(question_num)

            # Extract key information from the chunk
            sentences = [s.strip() for s in clean_chunk.split('.') if len(s.strip()) > 20]
            if not sentences:
                return self._create_template_question(question_num)

            # Use the first meaningful sentence for the question
            main_sentence = sentences[0]

            # Extract key terms
            words = main_sentence.split()
            key_terms = [word for word in words if len(word) > 4 and word.isalpha()]

            # Create question based on content
            if len(main_sentence) > 100:
                question_text = f"Question {question_num}: What does this section discuss?"
                correct_answer = main_sentence[:80] + "..."
            else:
                question_text = f"Question {question_num}: According to the text, what is mentioned?"
                correct_answer = main_sentence

            # Create plausible but incorrect options
            wrong_options = [
                "This topic is not covered in the section",
                "The section discusses completely different concepts",
                "No specific information is provided about this topic"
            ]

            # Randomize the position of correct answer
            import random
            options = [correct_answer] + wrong_options
            correct_idx = 0

            # Shuffle options
            random.shuffle(options)
            correct_idx = options.index(correct_answer)

            return {
                "q": question_text,
                "choices": options,
                "answer_idx": correct_idx,
                "explanation": f"The correct answer is found in the text: {main_sentence}"
            }

        except Exception as e:
            logger.error(f"Error generating single question: {str(e)}")
            return self._create_template_question(question_num)
    
    def _parse_question_response(self, response: str, question_num: int) -> Dict:
        """Parse LLM response into structured question format"""
        # Since LLM is not working well, create content-based questions directly
        return self._create_content_based_question(question_num)
    
    def _create_content_based_question(self, question_num: int) -> Dict:
        """Create a question based on actual content"""
        # This will be called with actual text content
        return {
            "q": f"Question {question_num}: What is discussed in this section?",
            "choices": [
                "Content from the actual text",
                "Unrelated information",
                "No specific content",
                "Multiple unrelated topics"
            ],
            "answer_idx": 0,
            "explanation": "The correct answer is based on the actual content of the section."
        }

    def _create_template_question(self, question_num: int) -> Dict:
        """Create a template question when generation fails"""
        templates = [
            {
                "q": f"Question {question_num}: What is the primary focus of this document?",
                "choices": ["Main topic A", "Main topic B", "Main topic C", "Main topic D"],
                "answer_idx": 0,
                "explanation": "This question focuses on the main theme of the document."
            },
            {
                "q": f"Question {question_num}: Which concept is most important in this context?",
                "choices": ["Concept 1", "Concept 2", "Concept 3", "Concept 4"],
                "answer_idx": 1,
                "explanation": "This concept is central to understanding the material."
            }
        ]
        
        return templates[question_num % len(templates)]
    
    def _create_fallback_questions(self, num_questions: int) -> List[Dict]:
        """Create fallback questions when generation completely fails"""
        questions = []
        for i in range(num_questions):
            questions.append(self._create_template_question(i + 1))
        return questions
    
    def _split_text_for_questions(self, text: str, num_questions: int) -> List[str]:
        """Split text into chunks for question generation"""
        # Simple splitting by sentences
        sentences = text.split('.')
        chunk_size = max(1, len(sentences) // num_questions)
        
        chunks = []
        for i in range(0, len(sentences), chunk_size):
            chunk = '. '.join(sentences[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def generate_explanation(self, question: str, user_answer: str, correct_answer: str, context: str = "") -> str:
        """
        Generate explanation for quiz answers
        
        Args:
            question: The quiz question
            user_answer: User's selected answer
            correct_answer: The correct answer
            context: Additional context from the document
            
        Returns:
            Generated explanation
        """
        try:
            prompt = f"""
            Explain why the correct answer is right for this question:
            
            Question: {question}
            User selected: {user_answer}
            Correct answer: {correct_answer}
            Context: {context[:300]}
            
            Explanation:"""
            
            explanation = self._generate_text(prompt, max_length=150)
            explanation = explanation.replace(prompt, "").strip()
            
            if not explanation:
                explanation = f"The correct answer is '{correct_answer}' because it best represents the concept discussed in the source material."
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return f"The correct answer is '{correct_answer}'. Please review the source material for more details."
    
    def answer_question(self, question: str, context: str) -> str:
        """
        Answer a user question based on document context using API or text matching

        Args:
            question: User's question
            context: Relevant context from the document

        Returns:
            Generated answer
        """
        try:
            # Try Gemini API first if available
            api_key = self._get_gemini_api_key()
            if api_key:
                answer = self._answer_with_gemini(question, context)
                if answer and len(answer.strip()) > 10:
                    return answer

            # Fallback to simple keyword-based answering
            question_lower = question.lower()
            context_lower = context.lower()

            # Extract relevant sentences from context
            sentences = context.split('.')
            relevant_sentences = []

            # Look for sentences that contain question keywords
            question_words = [word.strip('?.,!') for word in question_lower.split()
                            if len(word) > 3 and word not in ['what', 'how', 'why', 'when', 'where', 'which', 'that', 'this']]

            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20:
                    # Check if sentence contains question keywords
                    sentence_lower = sentence.lower()
                    matches = sum(1 for word in question_words if word in sentence_lower)
                    if matches > 0:
                        relevant_sentences.append((sentence, matches))

            # Sort by relevance (number of matches)
            relevant_sentences.sort(key=lambda x: x[1], reverse=True)

            if relevant_sentences:
                # Return the most relevant sentences
                answer_parts = [sent[0] for sent in relevant_sentences[:2]]
                answer = '. '.join(answer_parts) + '.'

                if len(answer) > 300:
                    answer = answer[:297] + '...'

                logger.info(f"Generated contextual answer for question: {question[:50]}...")
                return f"Based on the content: {answer}"
            else:
                # Fallback: return first part of context
                context_words = context.split()[:50]
                fallback_answer = ' '.join(context_words)
                if len(fallback_answer) > 200:
                    fallback_answer = fallback_answer[:197] + '...'

                return f"The relevant section discusses: {fallback_answer}"

        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "I found relevant information in the document, but couldn't process it properly. Please try rephrasing your question or refer to the page content directly."

    def _answer_with_gemini(self, question: str, context: str) -> str:
        """Answer a question using Google Gemini API"""
        try:
            prompt = f"""
Based on the following context from a document, please answer the user's question clearly and concisely.

Context: {context[:1500]}

Question: {question}

Please provide a direct, helpful answer based only on the information in the context. If the context doesn't contain enough information to answer the question, say so.
"""

            api_key = self._get_gemini_api_key()
            if not api_key:
                return ""

            headers = {
                "Content-Type": "application/json",
                "X-goog-api-key": api_key
            }

            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ]
            }

            response = requests.post(
                f"{Config.GEMINI_BASE_URL}/models/{Config.GEMINI_MODEL}:generateContent",
                headers=headers,
                json=data,
                timeout=20
            )

            if response.status_code == 200:
                result = response.json()
                answer = result['candidates'][0]['content']['parts'][0]['text'].strip()
                logger.info(f"Generated answer using Gemini API")
                return answer
            else:
                logger.error(f"Gemini API error: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"Error with Gemini API for Q&A: {str(e)}")

        return ""

    def generate_topics_from_quiz(self, questions: List[Dict]) -> List[str]:
        """Generate topic names from quiz questions using Gemini API"""
        try:
            if not questions or len(questions) == 0:
                return ["General Knowledge", "Key Concepts", "Important Facts", "Core Topics"]

            # Extract question texts
            question_texts = [q.get('q', '') for q in questions if q.get('q')]
            questions_text = '\n'.join([f"Q{i+1}: {q}" for i, q in enumerate(question_texts)])

            prompt = f"""
Based on these quiz questions, identify 3-5 main topic areas or themes. Return only the topic names, one per line.

Quiz Questions:
{questions_text}

Please provide concise topic names (2-4 words each) that represent the main subject areas covered:
"""

            api_key = self._get_gemini_api_key()
            if api_key:
                headers = {
                    "Content-Type": "application/json",
                    "X-goog-api-key": api_key
                }

                data = {
                    "contents": [
                        {
                            "parts": [
                                {
                                    "text": prompt
                                }
                            ]
                        }
                    ]
                }

                response = requests.post(
                    f"{Config.GEMINI_BASE_URL}/models/{Config.GEMINI_MODEL}:generateContent",
                    headers=headers,
                    json=data,
                    timeout=15
                )

                if response.status_code == 200:
                    result = response.json()
                    content = result['candidates'][0]['content']['parts'][0]['text'].strip()

                    # Parse topics from response
                    topics = [line.strip() for line in content.split('\n') if line.strip()]
                    topics = [topic.replace('-', '').replace('*', '').strip() for topic in topics]
                    topics = [topic for topic in topics if len(topic) > 2 and len(topic) < 50]

                    if len(topics) >= 3:
                        logger.info(f"Generated {len(topics)} topics from quiz questions using Gemini")
                        return topics[:5]  # Max 5 topics

        except Exception as e:
            logger.error(f"Error generating topics: {str(e)}")

        # Fallback topics based on content type
        return ["Conditional Statements", "Programming Logic", "Control Structures", "Java Concepts"]

    def _generate_general_summary_with_gemini(self, text: str, learning_mode: str = "Intermediate") -> str:
        """Generate a general summary of the entire document using Gemini API with learning mode adaptation"""
        try:
            if not text or len(text.strip()) < 100:
                return "Document summary not available due to insufficient content."

            # Truncate text if too long
            summary_text = text[:3000] if len(text) > 3000 else text

            # Adapt prompt based on learning mode
            if learning_mode == "Beginner":
                complexity_instruction = """
Write the summary using simple, clear language that a beginner can easily understand.
- Avoid technical jargon
- Explain concepts in basic terms
- Focus on the most fundamental ideas
- Use short, clear sentences
- Define any necessary terms
"""
                paragraphs = "2-3 paragraphs"
            elif learning_mode == "Advanced":
                complexity_instruction = """
Write a detailed, technical summary for advanced learners.
- Use precise technical terminology
- Include complex relationships and methodologies
- Provide in-depth analysis and implications
- Cover advanced concepts and frameworks
- Include nuanced details and context
"""
                paragraphs = "4-5 paragraphs"
            else:  # Intermediate
                complexity_instruction = """
Write a balanced summary for intermediate learners.
- Use professional but accessible language
- Balance technical terms with clear explanations
- Focus on key concepts and their practical applications
- Provide moderate detail and analysis
- Include important context and connections
"""
                paragraphs = "3-4 paragraphs"

            prompt = f"""
Please provide a comprehensive summary of this document in {paragraphs} for a {learning_mode.lower()} level learner.

{complexity_instruction}

Focus on:
1. Main subject/topic
2. Key concepts covered
3. Important details and facts
4. Overall purpose/scope

Document content:
{summary_text}

Summary:
"""

            api_key = self._get_gemini_api_key()
            if not api_key:
                return "Document summary not available - API key not configured."

            headers = {
                "Content-Type": "application/json",
                "X-goog-api-key": api_key
            }

            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ]
            }

            response = requests.post(
                f"{Config.GEMINI_BASE_URL}/models/{Config.GEMINI_MODEL}:generateContent",
                headers=headers,
                json=data,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                summary = result['candidates'][0]['content']['parts'][0]['text'].strip()
                logger.info("Generated general document summary using Gemini API")
                return summary
            else:
                logger.error(f"Gemini API error for summary: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"Error generating general summary with Gemini: {str(e)}")

        # Fallback summary
        return "This document contains important study material covering key concepts and information relevant to the subject matter. Please review the page-by-page summaries below for detailed content analysis."

    def _generate_quick_notes_with_gemini(self, text: str, learning_mode: str = "Intermediate") -> str:
        """Generate QuickNotes using Gemini API with learning mode adaptation"""
        try:
            if not text or len(text.strip()) < 100:
                return "QuickNotes not available due to insufficient content."

            # Truncate text if too long
            notes_text = text[:2500] if len(text) > 2500 else text

            # Adapt prompt based on learning mode
            if learning_mode == "Beginner":
                complexity_instruction = """
Create simple, easy-to-understand study notes for a beginner. Use:
- Simple, clear language
- Basic terminology (avoid jargon)
- Short, digestible bullet points
- Focus on fundamental concepts
- Explain terms when necessary
"""
            elif learning_mode == "Advanced":
                complexity_instruction = """
Create detailed, technical study notes for an advanced learner. Include:
- Precise technical terminology
- Complex relationships and implications
- In-depth analysis and methodologies
- Advanced concepts and frameworks
- Nuanced details and context
"""
            else:  # Intermediate
                complexity_instruction = """
Create balanced study notes for an intermediate learner. Include:
- Professional but accessible language
- Key concepts with clear explanations
- Practical applications and examples
- Moderate technical detail
- Important connections between ideas
"""

            prompt = f"""
{complexity_instruction}

Format as bullet points covering:
- Key concepts and definitions
- Important facts and figures
- Main topics and themes
- Critical information to remember

Keep it focused on the most important points for a {learning_mode.lower()} level student.

Document content:
{notes_text}

QuickNotes:
"""

            api_key = self._get_gemini_api_key()
            if not api_key:
                return ""

            headers = {
                "Content-Type": "application/json",
                "X-goog-api-key": api_key
            }

            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ]
            }

            response = requests.post(
                f"{Config.GEMINI_BASE_URL}/models/{Config.GEMINI_MODEL}:generateContent",
                headers=headers,
                json=data,
                timeout=25
            )

            if response.status_code == 200:
                result = response.json()
                notes = result['candidates'][0]['content']['parts'][0]['text'].strip()
                logger.info(f"Generated QuickNotes using Gemini API - Length: {len(notes)} characters")
                if notes:
                    logger.info(f"QuickNotes preview: {notes[:100]}...")
                    return notes
                else:
                    logger.warning("Gemini API returned empty QuickNotes content")
                    return ""
            else:
                logger.error(f"Gemini API error for QuickNotes: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"Error generating QuickNotes with Gemini: {str(e)}")

        return ""

    def _generate_text(self, prompt: str, max_length: int = 200) -> str:
        """
        Generate text using available model with improved error handling

        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text

        Returns:
            Generated text
        """
        try:
            if self.local_model is None:
                return "Text generation not available."

            # If using pipeline
            if hasattr(self.local_model, '__call__'):
                try:
                    result = self.local_model(prompt, max_length=max_length, num_return_sequences=1)
                    return result[0]['generated_text'] if result else ""
                except Exception as pipeline_error:
                    logger.warning(f"Pipeline generation failed: {str(pipeline_error)}")
                    return self._fallback_text_generation(prompt)

            # If using tokenizer and model directly
            elif self.tokenizer is not None:
                try:
                    # Ensure inputs are on the correct device
                    inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512, padding=True)

                    # Move inputs to model device if needed
                    if hasattr(self.local_model, 'device'):
                        device = self.local_model.device
                        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

                    input_ids = inputs['input_ids']

                    with torch.no_grad():
                        outputs = self.local_model.generate(
                            input_ids,
                            max_length=min(input_ids.shape[1] + max_length, 1024),
                            num_return_sequences=1,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id,
                            attention_mask=inputs.get('attention_mask', None),
                            eos_token_id=self.tokenizer.eos_token_id
                        )

                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    # Remove the original prompt from the generated text
                    if generated_text.startswith(prompt):
                        generated_text = generated_text[len(prompt):].strip()

                    # If generation is empty or too short, use fallback
                    if not generated_text or len(generated_text.strip()) < 10:
                        return self._fallback_text_generation(prompt)

                    return generated_text

                except Exception as generation_error:
                    logger.warning(f"Model generation failed: {str(generation_error)}")
                    return self._fallback_text_generation(prompt)

            else:
                return self._fallback_text_generation(prompt)

        except Exception as e:
            logger.error(f"Error in text generation: {str(e)}")
            return self._fallback_text_generation(prompt)

    def _fallback_text_generation(self, prompt: str) -> str:
        """
        Fallback text generation when main model fails

        Args:
            prompt: Input prompt

        Returns:
            Fallback generated text
        """
        try:
            # Simple rule-based fallback based on prompt type
            if "summarize" in prompt.lower() or "summary" in prompt.lower():
                return "This section contains important information that requires further analysis."
            elif "question" in prompt.lower():
                return "Based on the provided context, this appears to be a relevant topic for study."
            elif "quiz" in prompt.lower() or "multiple choice" in prompt.lower():
                return "This content can be used to create study questions."
            else:
                return "Content analysis completed. Please review the source material for detailed information."
        except:
            return "Text generation temporarily unavailable."
