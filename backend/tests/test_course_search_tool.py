"""
Tests for CourseSearchTool.execute method
"""
import unittest
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import CourseSearchTool
from test_fixtures import MockVectorStore, SEARCH_TEST_SCENARIOS, create_mock_search_results


class TestCourseSearchTool(unittest.TestCase):
    """Test cases for CourseSearchTool functionality"""
    
    def setUp(self):
        """Set up test fixtures before each test"""
        self.mock_vector_store = MockVectorStore()
        self.search_tool = CourseSearchTool(self.mock_vector_store)
    
    def test_successful_search_execution(self):
        """Test that execute method works with successful search results"""
        scenario = SEARCH_TEST_SCENARIOS['successful_search']
        
        # Configure mock to return test results
        self.mock_vector_store.mock_results = scenario['mock_results']
        
        # Execute the search
        result = self.search_tool.execute(query=scenario['query'])
        
        # Verify the result contains expected content
        for expected_text in scenario['expected_contains']:
            self.assertIn(expected_text, result, 
                         f"Expected '{expected_text}' in result: {result}")
        
        # Verify the vector store was called correctly
        self.assertEqual(len(self.mock_vector_store.search_calls), 1)
        call = self.mock_vector_store.search_calls[0]
        self.assertEqual(call['query'], scenario['query'])
        self.assertEqual(call['course_name'], None)
        self.assertEqual(call['lesson_number'], None)
    
    def test_empty_search_results(self):
        """Test behavior when search returns no results"""
        scenario = SEARCH_TEST_SCENARIOS['empty_search']
        
        # Configure mock to return empty results
        self.mock_vector_store.mock_results = scenario['mock_results']
        
        # Execute the search
        result = self.search_tool.execute(query=scenario['query'])
        
        # Verify appropriate message is returned
        for expected_text in scenario['expected_contains']:
            self.assertIn(expected_text, result,
                         f"Expected '{expected_text}' in result: {result}")
    
    def test_search_with_error(self):
        """Test behavior when search returns error"""
        scenario = SEARCH_TEST_SCENARIOS['search_with_error']
        
        # Configure mock to return error
        self.mock_vector_store.mock_results = scenario['mock_results']
        
        # Execute the search  
        result = self.search_tool.execute(query=scenario['query'])
        
        # Verify error message is returned
        for expected_text in scenario['expected_contains']:
            self.assertIn(expected_text, result,
                         f"Expected '{expected_text}' in result: {result}")
    
    def test_course_name_filtering(self):
        """Test search with course name filter"""
        scenario = SEARCH_TEST_SCENARIOS['course_filtered_search']
        
        # Configure mock to return filtered results
        self.mock_vector_store.mock_results = scenario['mock_results']
        
        # Execute the search with course filter
        result = self.search_tool.execute(
            query=scenario['query'],
            course_name=scenario['course_name']
        )
        
        # Verify the result contains expected content
        for expected_text in scenario['expected_contains']:
            self.assertIn(expected_text, result,
                         f"Expected '{expected_text}' in result: {result}")
        
        # Verify the vector store was called with course filter
        self.assertEqual(len(self.mock_vector_store.search_calls), 1)
        call = self.mock_vector_store.search_calls[0]
        self.assertEqual(call['course_name'], scenario['course_name'])
    
    def test_lesson_number_filtering(self):
        """Test search with lesson number filter"""
        scenario = SEARCH_TEST_SCENARIOS['lesson_filtered_search']
        
        # Configure mock to return filtered results
        self.mock_vector_store.mock_results = scenario['mock_results']
        
        # Execute the search with lesson filter
        result = self.search_tool.execute(
            query=scenario['query'],
            lesson_number=scenario['lesson_number']
        )
        
        # Verify the result contains expected content
        for expected_text in scenario['expected_contains']:
            self.assertIn(expected_text, result,
                         f"Expected '{expected_text}' in result: {result}")
        
        # Verify the vector store was called with lesson filter
        self.assertEqual(len(self.mock_vector_store.search_calls), 1)
        call = self.mock_vector_store.search_calls[0]
        self.assertEqual(call['lesson_number'], scenario['lesson_number'])
    
    def test_combined_filters(self):
        """Test search with both course name and lesson number filters"""
        # Configure mock with specific results
        mock_results = [create_mock_search_results(
            ["Specific lesson content for advanced topic"],
            [{"course_title": "Advanced Course", "lesson_number": 5}]
        )]
        self.mock_vector_store.mock_results = mock_results
        
        # Execute search with both filters
        result = self.search_tool.execute(
            query="advanced topic",
            course_name="Advanced Course", 
            lesson_number=5
        )
        
        # Verify both filters are present in result
        self.assertIn("Advanced Course", result)
        self.assertIn("Lesson 5", result)
        self.assertIn("advanced topic", result)
        
        # Verify vector store received both filters
        call = self.mock_vector_store.search_calls[0]
        self.assertEqual(call['course_name'], "Advanced Course")
        self.assertEqual(call['lesson_number'], 5)
    
    def test_sources_tracking(self):
        """Test that last_sources is properly tracked"""
        # Configure mock with multiple results
        mock_results = [create_mock_search_results(
            ["Content from lesson 1", "Content from lesson 2"],
            [
                {"course_title": "Test Course", "lesson_number": 1},
                {"course_title": "Test Course", "lesson_number": 2}
            ]
        )]
        self.mock_vector_store.mock_results = mock_results
        
        # Execute search
        result = self.search_tool.execute(query="test query")
        
        # Verify sources are tracked
        expected_sources = ["Test Course - Lesson 1", "Test Course - Lesson 2"]
        self.assertEqual(self.search_tool.last_sources, expected_sources)
    
    def test_sources_without_lesson_number(self):
        """Test sources tracking when lesson number is not available"""
        # Configure mock with results missing lesson numbers
        mock_results = [create_mock_search_results(
            ["Course content without lesson"],
            [{"course_title": "Test Course"}]  # No lesson_number
        )]
        self.mock_vector_store.mock_results = mock_results
        
        # Execute search
        result = self.search_tool.execute(query="test query")
        
        # Verify source only contains course name
        expected_sources = ["Test Course"]
        self.assertEqual(self.search_tool.last_sources, expected_sources)
    
    def test_tool_definition(self):
        """Test that tool definition is properly formatted"""
        definition = self.search_tool.get_tool_definition()
        
        # Verify required fields
        self.assertEqual(definition['name'], 'search_course_content')
        self.assertIn('description', definition)
        self.assertIn('input_schema', definition)
        
        # Verify input schema structure
        schema = definition['input_schema']
        self.assertEqual(schema['type'], 'object')
        self.assertIn('properties', schema)
        self.assertIn('required', schema)
        
        # Verify required query parameter
        self.assertIn('query', schema['required'])
        self.assertIn('query', schema['properties'])
        
        # Verify optional parameters
        properties = schema['properties']
        self.assertIn('course_name', properties)
        self.assertIn('lesson_number', properties)


if __name__ == '__main__':
    # Print a header to clearly identify test output
    print("=" * 60)
    print("TESTING CourseSearchTool.execute method")
    print("=" * 60)
    
    unittest.main(verbosity=2)