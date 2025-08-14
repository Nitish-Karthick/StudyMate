"""
Quiz Management Module for StudyMate.ai
Handles quiz generation, answer validation, scoring, and progress tracking
"""

import json
import time
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuizManager:
    """Manages quiz functionality including generation, scoring, and progress tracking"""
    
    def __init__(self):
        """Initialize quiz manager"""
        self.current_quiz = None
        self.user_answers = []
        self.start_time = None
        self.end_time = None
        self.current_question_index = 0
    
    def create_quiz(self, questions: List[Dict], quiz_title: str = "Study Quiz") -> Dict:
        """
        Create a new quiz from generated questions
        
        Args:
            questions: List of question dictionaries
            quiz_title: Title for the quiz
            
        Returns:
            Quiz data structure
        """
        try:
            quiz_data = {
                'title': quiz_title,
                'questions': questions,
                'total_questions': len(questions),
                'created_at': datetime.now().isoformat(),
                'quiz_id': f"quiz_{int(time.time())}"
            }
            
            self.current_quiz = quiz_data
            self.user_answers = []
            self.current_question_index = 0
            self.start_time = None
            self.end_time = None
            
            logger.info(f"Created quiz '{quiz_title}' with {len(questions)} questions")
            return quiz_data
            
        except Exception as e:
            logger.error(f"Error creating quiz: {str(e)}")
            raise Exception(f"Failed to create quiz: {str(e)}")
    
    def start_quiz(self) -> Dict:
        """
        Start the quiz session
        
        Returns:
            Quiz session data
        """
        if not self.current_quiz:
            raise ValueError("No quiz available. Create a quiz first.")
        
        self.start_time = time.time()
        self.user_answers = []
        self.current_question_index = 0
        
        session_data = {
            'quiz_id': self.current_quiz['quiz_id'],
            'started_at': datetime.now().isoformat(),
            'current_question': 0,
            'total_questions': self.current_quiz['total_questions']
        }
        
        logger.info(f"Started quiz session: {self.current_quiz['quiz_id']}")
        return session_data
    
    def get_current_question(self) -> Optional[Dict]:
        """
        Get the current question in the quiz
        
        Returns:
            Current question data or None if quiz is complete
        """
        if not self.current_quiz:
            return None
        
        if self.current_question_index >= len(self.current_quiz['questions']):
            return None
        
        question = self.current_quiz['questions'][self.current_question_index].copy()
        question['question_number'] = self.current_question_index + 1
        question['total_questions'] = self.current_quiz['total_questions']
        
        return question
    
    def submit_answer(self, selected_choice_index: int) -> Dict:
        """
        Submit an answer for the current question
        
        Args:
            selected_choice_index: Index of the selected choice
            
        Returns:
            Answer result with feedback
        """
        try:
            current_question = self.get_current_question()
            if not current_question:
                raise ValueError("No current question available")
            
            correct_answer_index = current_question['answer_idx']
            is_correct = selected_choice_index == correct_answer_index
            
            # Record the answer
            answer_record = {
                'question_index': self.current_question_index,
                'question': current_question['q'],
                'selected_choice_index': selected_choice_index,
                'selected_choice': current_question['choices'][selected_choice_index],
                'correct_choice_index': correct_answer_index,
                'correct_choice': current_question['choices'][correct_answer_index],
                'is_correct': is_correct,
                'explanation': current_question.get('explanation', ''),
                'answered_at': datetime.now().isoformat()
            }
            
            self.user_answers.append(answer_record)
            
            # Prepare feedback
            feedback = {
                'is_correct': is_correct,
                'selected_answer': current_question['choices'][selected_choice_index],
                'correct_answer': current_question['choices'][correct_answer_index],
                'explanation': current_question.get('explanation', ''),
                'question_number': self.current_question_index + 1,
                'total_questions': self.current_quiz['total_questions']
            }
            
            logger.info(f"Answer submitted for question {self.current_question_index + 1}: {'Correct' if is_correct else 'Incorrect'}")
            return feedback
            
        except Exception as e:
            logger.error(f"Error submitting answer: {str(e)}")
            raise Exception(f"Failed to submit answer: {str(e)}")
    
    def next_question(self) -> bool:
        """
        Move to the next question
        
        Returns:
            True if there's a next question, False if quiz is complete
        """
        self.current_question_index += 1
        
        if self.current_question_index >= len(self.current_quiz['questions']):
            self.end_time = time.time()
            logger.info("Quiz completed")
            return False
        
        return True
    
    def get_quiz_progress(self) -> Dict:
        """
        Get current quiz progress
        
        Returns:
            Progress information
        """
        if not self.current_quiz:
            return {'progress': 0, 'current': 0, 'total': 0}
        
        total_questions = self.current_quiz['total_questions']
        answered_questions = len(self.user_answers)
        progress_percentage = (answered_questions / total_questions) * 100 if total_questions > 0 else 0
        
        return {
            'progress': progress_percentage,
            'current': answered_questions,
            'total': total_questions,
            'current_question_index': self.current_question_index
        }
    
    def calculate_final_score(self) -> Dict:
        """
        Calculate final quiz score and statistics
        
        Returns:
            Complete score report
        """
        if not self.current_quiz or not self.user_answers:
            return {'error': 'No quiz data available for scoring'}
        
        total_questions = len(self.current_quiz['questions'])
        correct_answers = sum(1 for answer in self.user_answers if answer['is_correct'])
        incorrect_answers = total_questions - correct_answers
        
        score_percentage = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
        
        # Calculate time taken
        time_taken = 0
        if self.start_time and self.end_time:
            time_taken = self.end_time - self.start_time
        
        # Determine performance level
        if score_percentage >= 90:
            performance_level = "Excellent"
        elif score_percentage >= 80:
            performance_level = "Good"
        elif score_percentage >= 70:
            performance_level = "Average"
        elif score_percentage >= 60:
            performance_level = "Below Average"
        else:
            performance_level = "Needs Improvement"
        
        score_report = {
            'quiz_id': self.current_quiz['quiz_id'],
            'quiz_title': self.current_quiz['title'],
            'total_questions': total_questions,
            'correct_answers': correct_answers,
            'incorrect_answers': incorrect_answers,
            'score_percentage': round(score_percentage, 1),
            'performance_level': performance_level,
            'time_taken_seconds': round(time_taken, 1),
            'time_taken_formatted': self._format_time(time_taken),
            'completed_at': datetime.now().isoformat(),
            'detailed_answers': self.user_answers
        }
        
        logger.info(f"Quiz completed with score: {score_percentage:.1f}% ({correct_answers}/{total_questions})")
        return score_report
    
    def get_incorrect_answers(self) -> List[Dict]:
        """
        Get all incorrect answers for review
        
        Returns:
            List of incorrect answers with explanations
        """
        incorrect_answers = []
        
        for answer in self.user_answers:
            if not answer['is_correct']:
                incorrect_answers.append({
                    'question': answer['question'],
                    'your_answer': answer['selected_choice'],
                    'correct_answer': answer['correct_choice'],
                    'explanation': answer['explanation'],
                    'question_number': answer['question_index'] + 1
                })
        
        return incorrect_answers
    
    def get_topic_performance(self, real_topics: List[str] = None) -> Dict:
        """
        Analyze performance by topics using real topic names

        Args:
            real_topics: List of actual topic names from content analysis

        Returns:
            Topic-based performance analysis
        """
        total_questions = len(self.user_answers)
        if total_questions == 0:
            return {}

        # Use real topics if provided, otherwise fallback to generic ones
        if real_topics and len(real_topics) > 0:
            topic_names = real_topics[:4]  # Use up to 4 topics
        else:
            topic_names = ["Conditional Statements", "Programming Logic", "Control Structures", "Java Concepts"]

        # Create topic groups
        topics = {}
        for topic in topic_names:
            topics[topic] = {'correct': 0, 'total': 0}

        # Distribute questions across topics
        for i, answer in enumerate(self.user_answers):
            topic = topic_names[i % len(topic_names)]
            topics[topic]['total'] += 1
            if answer['is_correct']:
                topics[topic]['correct'] += 1

        # Calculate percentages
        topic_performance = {}
        for topic, data in topics.items():
            if data['total'] > 0:
                percentage = (data['correct'] / data['total']) * 100
                topic_performance[topic] = round(percentage, 1)
            else:
                topic_performance[topic] = 0

        return topic_performance
    
    def reset_quiz(self):
        """Reset the current quiz session"""
        self.user_answers = []
        self.current_question_index = 0
        self.start_time = None
        self.end_time = None
        
        logger.info("Quiz session reset")
    
    def _format_time(self, seconds: float) -> str:
        """
        Format time in seconds to human-readable format
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string
        """
        if seconds < 60:
            return f"{int(seconds)} seconds"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            remaining_seconds = int(seconds % 60)
            return f"{minutes}m {remaining_seconds}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def export_quiz_data(self) -> Dict:
        """
        Export complete quiz data for analysis or storage
        
        Returns:
            Complete quiz session data
        """
        if not self.current_quiz:
            return {}
        
        export_data = {
            'quiz_metadata': self.current_quiz,
            'user_answers': self.user_answers,
            'session_info': {
                'start_time': self.start_time,
                'end_time': self.end_time,
                'current_question_index': self.current_question_index
            },
            'score_report': self.calculate_final_score() if self.end_time else None,
            'exported_at': datetime.now().isoformat()
        }
        
        return export_data
    
    def get_quiz_summary(self) -> Dict:
        """
        Get a summary of the current quiz state
        
        Returns:
            Quiz summary information
        """
        if not self.current_quiz:
            return {'status': 'No quiz loaded'}
        
        progress = self.get_quiz_progress()
        
        summary = {
            'quiz_title': self.current_quiz['title'],
            'total_questions': self.current_quiz['total_questions'],
            'questions_answered': len(self.user_answers),
            'current_question': self.current_question_index + 1,
            'progress_percentage': progress['progress'],
            'is_completed': self.current_question_index >= self.current_quiz['total_questions'],
            'quiz_started': self.start_time is not None
        }
        
        if summary['is_completed'] and self.user_answers:
            correct_count = sum(1 for answer in self.user_answers if answer['is_correct'])
            summary['final_score'] = (correct_count / len(self.user_answers)) * 100
        
        return summary
