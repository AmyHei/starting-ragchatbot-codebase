"""
Test fixtures and mock data for RAG system testing
"""
from typing import List, Dict, Any
from unittest.mock import Mock, MagicMock
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Course, Lesson, CourseChunk
from vector_store import SearchResults

class MockVectorStore:
    """Mock VectorStore for testing"""
    
    def __init__(self, mock_results: List[SearchResults] = None):
        self.mock_results = mock_results or []
        self.search_calls = []
        self.current_result_index = 0
        self._resolve_course_name_calls = []
        self._resolve_course_name_return = None
        
    def search(self, query: str, course_name: str = None, lesson_number: int = None) -> SearchResults:
        """Mock search method that tracks calls and returns predefined results"""
        self.search_calls.append({
            'query': query,
            'course_name': course_name, 
            'lesson_number': lesson_number
        })
        
        if self.mock_results and self.current_result_index < len(self.mock_results):
            result = self.mock_results[self.current_result_index]
            self.current_result_index += 1
            return result
        
        # Default empty result
        return SearchResults.empty("No results configured")
    
    def _resolve_course_name(self, course_name: str):
        """Mock course name resolution"""
        self._resolve_course_name_calls.append(course_name)
        return self._resolve_course_name_return

def create_mock_search_results(documents: List[str], metadata: List[Dict[str, Any]] = None, 
                              distances: List[float] = None, error: str = None) -> SearchResults:
    """Helper to create SearchResults for testing"""
    if metadata is None:
        metadata = [{"course_title": "Test Course", "lesson_number": 1} for _ in documents]
    if distances is None:
        distances = [0.1 * i for i in range(len(documents))]
    
    return SearchResults(
        documents=documents,
        metadata=metadata,
        distances=distances,
        error=error
    )

def create_sample_course() -> Course:
    """Create a sample course for testing"""
    lessons = [
        Lesson(lesson_number=1, title="Introduction", content="Welcome to the course", lesson_link="http://example.com/lesson1"),
        Lesson(lesson_number=2, title="Advanced Topics", content="Deep dive into advanced concepts", lesson_link="http://example.com/lesson2")
    ]
    
    return Course(
        title="Test Course",
        instructor="Test Instructor", 
        course_link="http://example.com/course",
        lessons=lessons
    )

def create_sample_course_chunks() -> List[CourseChunk]:
    """Create sample course chunks for testing"""
    return [
        CourseChunk(
            content="Introduction to the course material",
            course_title="Test Course",
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Advanced concepts and implementations", 
            course_title="Test Course",
            lesson_number=2,
            chunk_index=1
        )
    ]

# Test scenarios for CourseSearchTool
SEARCH_TEST_SCENARIOS = {
    'successful_search': {
        'description': 'Search returns relevant results',
        'mock_results': [create_mock_search_results(
            ["This is content about Python programming", "More details about functions"], 
            [{"course_title": "Python Basics", "lesson_number": 1}, {"course_title": "Python Basics", "lesson_number": 2}]
        )],
        'query': 'Python programming',
        'expected_contains': ['Python Basics', 'Python programming', 'More details']
    },
    
    'empty_search': {
        'description': 'Search returns no results', 
        'mock_results': [SearchResults.empty("")],
        'query': 'nonexistent topic',
        'expected_contains': ['No relevant content found']
    },
    
    'search_with_error': {
        'description': 'Search returns error',
        'mock_results': [SearchResults.empty("Database connection failed")], 
        'query': 'any query',
        'expected_contains': ['Database connection failed']
    },
    
    'course_filtered_search': {
        'description': 'Search with course name filter',
        'mock_results': [create_mock_search_results(
            ["Content specific to Machine Learning"],
            [{"course_title": "ML Course", "lesson_number": 1}]
        )],
        'query': 'neural networks',
        'course_name': 'ML Course',
        'expected_contains': ['ML Course', 'Content specific to Machine Learning']
    },
    
    'lesson_filtered_search': {
        'description': 'Search with lesson number filter', 
        'mock_results': [create_mock_search_results(
            ["Lesson 3 specific content"],
            [{"course_title": "Test Course", "lesson_number": 3}]
        )],
        'query': 'specific topic',
        'lesson_number': 3,
        'expected_contains': ['Lesson 3', 'Test Course', 'specific topic']
    }
}

# Mock Anthropic response for AI Generator testing
MOCK_ANTHROPIC_RESPONSES = {
    'direct_response': Mock(
        stop_reason='end_turn',
        content=[Mock(text='This is a direct response without tool use')]
    ),
    
    'tool_use_response': Mock(
        stop_reason='tool_use',
        content=[
            Mock(
                type='tool_use',
                name='search_course_content',
                id='tool_call_123',
                input={'query': 'Python programming', 'course_name': 'Python Basics'}
            )
        ]
    ),
    
    'final_response_after_tool': Mock(
        stop_reason='end_turn', 
        content=[Mock(text='Based on the search results, Python is a programming language...')]
    )
}