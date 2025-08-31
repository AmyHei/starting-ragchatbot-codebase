"""
Tests for AIGenerator tool calling functionality
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator
from search_tools import ToolManager, CourseSearchTool
from test_fixtures import MockVectorStore, MOCK_ANTHROPIC_RESPONSES


class TestAIGenerator(unittest.TestCase):
    """Test cases for AIGenerator tool calling functionality"""
    
    def setUp(self):
        """Set up test fixtures before each test"""
        self.api_key = "test_api_key"
        self.model = "claude-sonnet-4-20250514"
        
        # Create mock tool manager
        self.mock_vector_store = MockVectorStore()
        self.tool_manager = ToolManager()
        self.search_tool = CourseSearchTool(self.mock_vector_store)
        self.tool_manager.register_tool(self.search_tool)
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_direct_response_without_tools(self, mock_anthropic_class):
        """Test direct response when no tools are needed"""
        # Setup mocks
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.return_value = MOCK_ANTHROPIC_RESPONSES['direct_response']
        
        ai_generator = AIGenerator(self.api_key, self.model)
        
        # Test query without tools
        result = ai_generator.generate_response(
            query="What is 2+2?",
            tools=None,
            tool_manager=None
        )
        
        # Verify direct response
        self.assertEqual(result, "This is a direct response without tool use")
        
        # Verify API was called correctly
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args[1]
        self.assertNotIn('tools', call_args)
        self.assertNotIn('tool_choice', call_args)
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_execution_flow(self, mock_anthropic_class):
        """Test complete tool execution flow"""
        # Setup mocks
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # First call returns tool use, second call returns final response
        mock_client.messages.create.side_effect = [
            MOCK_ANTHROPIC_RESPONSES['tool_use_response'],
            MOCK_ANTHROPIC_RESPONSES['final_response_after_tool']
        ]
        
        # Setup mock vector store to return results
        self.mock_vector_store.mock_results = [
            Mock(error=None, is_empty=lambda: False)
        ]
        
        # Mock the search tool's execute method
        with patch.object(self.search_tool, 'execute', return_value="Python is a programming language"):
            ai_generator = AIGenerator(self.api_key, self.model)
            
            # Test query that triggers tool use
            result = ai_generator.generate_response(
                query="Tell me about Python programming",
                tools=self.tool_manager.get_tool_definitions(),
                tool_manager=self.tool_manager
            )
            
            # Verify final response after tool execution
            self.assertEqual(result, "Based on the search results, Python is a programming language...")
            
            # Verify two API calls were made (initial + follow-up)
            self.assertEqual(mock_client.messages.create.call_count, 2)
            
            # Verify tools were included in first call
            first_call = mock_client.messages.create.call_args_list[0][1]
            self.assertIn('tools', first_call)
            self.assertIn('tool_choice', first_call)
            self.assertEqual(first_call['tool_choice']['type'], 'auto')
    
    @patch('ai_generator.anthropic.Anthropic') 
    def test_tool_execution_parameters(self, mock_anthropic_class):
        """Test that tool execution receives correct parameters"""
        # Setup mocks
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.side_effect = [
            MOCK_ANTHROPIC_RESPONSES['tool_use_response'],
            MOCK_ANTHROPIC_RESPONSES['final_response_after_tool']
        ]
        
        # Mock tool execution to track parameters
        execute_calls = []
        def mock_execute(**kwargs):
            execute_calls.append(kwargs)
            return "Mock search result"
        
        with patch.object(self.search_tool, 'execute', side_effect=mock_execute):
            ai_generator = AIGenerator(self.api_key, self.model)
            
            result = ai_generator.generate_response(
                query="Search for Python in ML course",
                tools=self.tool_manager.get_tool_definitions(),
                tool_manager=self.tool_manager
            )
            
            # Verify tool was executed with correct parameters
            self.assertEqual(len(execute_calls), 1)
            call_params = execute_calls[0]
            self.assertEqual(call_params['query'], 'Python programming')
            self.assertEqual(call_params['course_name'], 'Python Basics')
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_conversation_history_inclusion(self, mock_anthropic_class):
        """Test that conversation history is included in system prompt"""
        # Setup mocks
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.return_value = MOCK_ANTHROPIC_RESPONSES['direct_response']
        
        ai_generator = AIGenerator(self.api_key, self.model)
        
        # Test with conversation history
        conversation_history = "Previous conversation context"
        result = ai_generator.generate_response(
            query="Follow up question",
            conversation_history=conversation_history,
            tools=None,
            tool_manager=None
        )
        
        # Verify history was included in system prompt
        call_args = mock_client.messages.create.call_args[1]
        system_content = call_args['system']
        self.assertIn(conversation_history, system_content)
        self.assertIn("Previous conversation:", system_content)
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_execution_error_handling(self, mock_anthropic_class):
        """Test error handling during tool execution"""
        # Setup mocks
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.side_effect = [
            MOCK_ANTHROPIC_RESPONSES['tool_use_response'],
            MOCK_ANTHROPIC_RESPONSES['final_response_after_tool']
        ]
        
        # Mock tool execution to raise an error
        with patch.object(self.search_tool, 'execute', side_effect=Exception("Tool execution failed")):
            ai_generator = AIGenerator(self.api_key, self.model)
            
            # This should still work - the tool manager should handle errors gracefully
            try:
                result = ai_generator.generate_response(
                    query="Search query",
                    tools=self.tool_manager.get_tool_definitions(),
                    tool_manager=self.tool_manager
                )
                # If we get here, the error was handled properly
                self.assertIsInstance(result, str)
            except Exception as e:
                # If an exception propagated, that might indicate an issue
                self.fail(f"Tool execution error was not handled gracefully: {e}")
    
    def test_tool_definitions_integration(self):
        """Test that tool definitions are properly formatted for Anthropic API"""
        ai_generator = AIGenerator(self.api_key, self.model)
        tool_definitions = self.tool_manager.get_tool_definitions()
        
        # Verify we have the expected tools
        self.assertEqual(len(tool_definitions), 1)
        search_tool_def = tool_definitions[0]
        
        # Verify tool definition structure
        self.assertEqual(search_tool_def['name'], 'search_course_content')
        self.assertIn('description', search_tool_def)
        self.assertIn('input_schema', search_tool_def)
        
        # Verify schema is valid for Anthropic
        schema = search_tool_def['input_schema']
        self.assertEqual(schema['type'], 'object')
        self.assertIn('properties', schema)
        self.assertIn('required', schema)
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_api_parameters_configuration(self, mock_anthropic_class):
        """Test that API parameters are configured correctly"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.return_value = MOCK_ANTHROPIC_RESPONSES['direct_response']
        
        ai_generator = AIGenerator(self.api_key, self.model)
        
        # Test basic configuration
        result = ai_generator.generate_response(query="test query")
        
        # Verify API call parameters
        call_args = mock_client.messages.create.call_args[1]
        self.assertEqual(call_args['model'], self.model)
        self.assertEqual(call_args['temperature'], 0)
        self.assertEqual(call_args['max_tokens'], 800)
        self.assertIn('messages', call_args)
        self.assertIn('system', call_args)
        
        # Verify message structure
        messages = call_args['messages']
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]['role'], 'user')
        self.assertEqual(messages[0]['content'], 'test query')


class TestAIGeneratorIntegration(unittest.TestCase):
    """Integration tests for AIGenerator with real tool manager"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.api_key = "test_api_key" 
        self.model = "claude-sonnet-4-20250514"
        
        # Create real tool manager with mock vector store
        self.mock_vector_store = MockVectorStore()
        self.tool_manager = ToolManager()
        self.search_tool = CourseSearchTool(self.mock_vector_store)
        self.tool_manager.register_tool(self.search_tool)
    
    def test_tool_manager_execute_tool(self):
        """Test that tool manager can execute tools correctly"""
        # Setup mock results
        from test_fixtures import create_mock_search_results
        mock_result = create_mock_search_results(
            ["Python is a programming language"],
            [{"course_title": "Python Course", "lesson_number": 1}]
        )
        self.mock_vector_store.mock_results = [mock_result]
        
        # Execute tool through manager
        result = self.tool_manager.execute_tool(
            "search_course_content",
            query="Python programming",
            course_name="Python Course"
        )
        
        # Verify result
        self.assertIn("Python is a programming language", result)
        self.assertIn("Python Course", result)
        
        # Verify vector store was called
        self.assertEqual(len(self.mock_vector_store.search_calls), 1)
        call = self.mock_vector_store.search_calls[0]
        self.assertEqual(call['query'], "Python programming")
        self.assertEqual(call['course_name'], "Python Course")
    
    def test_sources_tracking_integration(self):
        """Test that sources are properly tracked through the tool manager"""
        from test_fixtures import create_mock_search_results
        mock_result = create_mock_search_results(
            ["Content 1", "Content 2"],
            [
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2}
            ]
        )
        self.mock_vector_store.mock_results = [mock_result]
        
        # Execute search
        self.tool_manager.execute_tool("search_course_content", query="test")
        
        # Verify sources are tracked
        sources = self.tool_manager.get_last_sources()
        expected_sources = ["Course A - Lesson 1", "Course B - Lesson 2"]
        self.assertEqual(sources, expected_sources)
        
        # Verify reset works
        self.tool_manager.reset_sources()
        self.assertEqual(self.tool_manager.get_last_sources(), [])


if __name__ == '__main__':
    print("=" * 60)
    print("TESTING AIGenerator tool calling functionality")  
    print("=" * 60)
    
    unittest.main(verbosity=2)