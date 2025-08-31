"""
Tests for RAG system query handling
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSystem
from config import Config
from test_fixtures import MockVectorStore, MOCK_ANTHROPIC_RESPONSES, create_mock_search_results


class MockConfig:
    """Mock configuration for testing"""
    ANTHROPIC_API_KEY = "test_key"
    ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    MAX_RESULTS = 5  # Note: This is different from the real config which has 0!
    MAX_HISTORY = 2
    CHROMA_PATH = "./test_chroma_db"


class TestRAGSystem(unittest.TestCase):
    """Test cases for RAG system query handling"""
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def setUp(self, mock_session, mock_doc_proc, mock_ai_gen, mock_vector_store):
        """Set up test fixtures with mocked dependencies"""
        self.mock_config = MockConfig()
        
        # Setup mock instances
        self.mock_vector_store_instance = Mock()
        mock_vector_store.return_value = self.mock_vector_store_instance
        
        self.mock_ai_generator_instance = Mock()
        mock_ai_gen.return_value = self.mock_ai_generator_instance
        
        self.mock_doc_processor_instance = Mock()
        mock_doc_proc.return_value = self.mock_doc_processor_instance
        
        self.mock_session_manager_instance = Mock()
        mock_session.return_value = self.mock_session_manager_instance
        
        # Create RAG system
        self.rag_system = RAGSystem(self.mock_config)
    
    def test_successful_query_processing(self):
        """Test successful query processing with tool execution"""
        # Setup mocks
        self.mock_session_manager_instance.get_conversation_history.return_value = None
        self.mock_ai_generator_instance.generate_response.return_value = "Python is a programming language used for AI development."
        
        # Mock sources tracking
        self.rag_system.tool_manager.get_last_sources = Mock(return_value=["Python Course - Lesson 1"])
        self.rag_system.tool_manager.reset_sources = Mock()
        
        # Execute query
        response, sources = self.rag_system.query("What is Python?")
        
        # Verify response
        self.assertEqual(response, "Python is a programming language used for AI development.")
        self.assertEqual(sources, ["Python Course - Lesson 1"])
        
        # Verify AI generator was called with correct parameters
        self.mock_ai_generator_instance.generate_response.assert_called_once()
        call_args = self.mock_ai_generator_instance.generate_response.call_args[1]
        self.assertIn("What is Python?", call_args['query'])
        self.assertIsNotNone(call_args['tools'])
        self.assertIsNotNone(call_args['tool_manager'])
        
        # Verify sources were reset
        self.rag_system.tool_manager.reset_sources.assert_called_once()
    
    def test_query_with_session_history(self):
        """Test query processing with conversation history"""
        # Setup mocks
        conversation_history = "User: Hello\nAssistant: Hi there!"
        self.mock_session_manager_instance.get_conversation_history.return_value = conversation_history
        self.mock_ai_generator_instance.generate_response.return_value = "Follow-up response"
        
        # Mock sources
        self.rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        self.rag_system.tool_manager.reset_sources = Mock()
        
        # Execute query with session ID
        response, sources = self.rag_system.query("Follow up question", session_id="test_session")
        
        # Verify history was retrieved and used
        self.mock_session_manager_instance.get_conversation_history.assert_called_with("test_session")
        
        call_args = self.mock_ai_generator_instance.generate_response.call_args[1]
        self.assertEqual(call_args['conversation_history'], conversation_history)
        
        # Verify session was updated
        self.mock_session_manager_instance.add_exchange.assert_called_with("test_session", "Follow up question", "Follow-up response")
    
    def test_query_without_session(self):
        """Test query processing without session ID"""
        # Setup mocks
        self.mock_ai_generator_instance.generate_response.return_value = "Direct response"
        self.rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        self.rag_system.tool_manager.reset_sources = Mock()
        
        # Execute query without session
        response, sources = self.rag_system.query("Direct question")
        
        # Verify no session methods were called
        self.mock_session_manager_instance.get_conversation_history.assert_not_called()
        self.mock_session_manager_instance.add_exchange.assert_not_called()
        
        # Verify AI generator was called with no history
        call_args = self.mock_ai_generator_instance.generate_response.call_args[1]
        self.assertIsNone(call_args['conversation_history'])
    
    def test_tool_definitions_passed_to_ai(self):
        """Test that tool definitions are correctly passed to AI generator"""
        # Setup mocks
        self.mock_ai_generator_instance.generate_response.return_value = "Response"
        self.rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        self.rag_system.tool_manager.reset_sources = Mock()
        
        # Execute query
        response, sources = self.rag_system.query("Test query")
        
        # Verify tools were passed
        call_args = self.mock_ai_generator_instance.generate_response.call_args[1]
        tools = call_args['tools']
        tool_manager = call_args['tool_manager']
        
        # Verify we have tools
        self.assertIsNotNone(tools)
        self.assertIsNotNone(tool_manager)
        
        # Verify it's the same tool manager
        self.assertEqual(tool_manager, self.rag_system.tool_manager)
    
    def test_prompt_formatting(self):
        """Test that query is properly formatted as a prompt"""
        # Setup mocks
        self.mock_ai_generator_instance.generate_response.return_value = "Response"
        self.rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        self.rag_system.tool_manager.reset_sources = Mock()
        
        user_query = "What is machine learning?"
        
        # Execute query
        response, sources = self.rag_system.query(user_query)
        
        # Verify query was formatted correctly
        call_args = self.mock_ai_generator_instance.generate_response.call_args[1]
        formatted_query = call_args['query']
        
        self.assertIn(user_query, formatted_query)
        self.assertIn("Answer this question about course materials:", formatted_query)


class TestRAGSystemIntegration(unittest.TestCase):
    """Integration tests with real components where possible"""
    
    def test_config_max_results_issue(self):
        """Test that identifies the MAX_RESULTS = 0 configuration issue"""
        # Load real config to test the issue
        from config import config as real_config
        
        # This test specifically checks for the configuration issue
        self.assertEqual(real_config.MAX_RESULTS, 0, 
                        "MAX_RESULTS is set to 0, which will cause search to return no results!")
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator') 
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_real_tool_manager_setup(self, mock_session, mock_doc_proc, mock_ai_gen, mock_vector_store):
        """Test that tool manager is properly set up with real tools"""
        mock_config = MockConfig()
        
        # Mock the dependencies
        mock_vector_store.return_value = Mock()
        mock_ai_gen.return_value = Mock()
        mock_doc_proc.return_value = Mock()
        mock_session.return_value = Mock()
        
        # Create RAG system
        rag_system = RAGSystem(mock_config)
        
        # Verify tool manager has the expected tools
        tool_definitions = rag_system.tool_manager.get_tool_definitions()
        
        # Should have 2 tools: search and outline
        self.assertEqual(len(tool_definitions), 2)
        
        tool_names = [tool['name'] for tool in tool_definitions]
        self.assertIn('search_course_content', tool_names)
        self.assertIn('get_course_outline', tool_names)
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_vector_store_initialization_parameters(self, mock_session, mock_doc_proc, mock_ai_gen, mock_vector_store):
        """Test that VectorStore is initialized with correct parameters from config"""
        mock_config = MockConfig()
        mock_config.MAX_RESULTS = 5  # Set to non-zero value
        
        # Create RAG system
        rag_system = RAGSystem(mock_config)
        
        # Verify VectorStore was initialized with correct parameters
        mock_vector_store.assert_called_once_with(
            mock_config.CHROMA_PATH,
            mock_config.EMBEDDING_MODEL, 
            mock_config.MAX_RESULTS
        )
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_ai_generator_initialization_parameters(self, mock_session, mock_doc_proc, mock_ai_gen, mock_vector_store):
        """Test that AIGenerator is initialized with correct parameters"""
        mock_config = MockConfig()
        
        # Create RAG system
        rag_system = RAGSystem(mock_config)
        
        # Verify AIGenerator was initialized with correct parameters
        mock_ai_gen.assert_called_once_with(
            mock_config.ANTHROPIC_API_KEY,
            mock_config.ANTHROPIC_MODEL
        )


class TestRAGSystemRealConfigIssues(unittest.TestCase):
    """Tests that specifically identify real configuration issues"""
    
    def test_real_config_max_results_zero(self):
        """This test will FAIL and identify the MAX_RESULTS=0 issue"""
        from config import config as real_config
        
        # This assertion will fail and clearly show the issue
        self.assertGreater(real_config.MAX_RESULTS, 0, 
                          "CRITICAL CONFIG ISSUE: MAX_RESULTS is set to 0 in config.py line 24. "
                          "This causes vector searches to return 0 results, explaining why "
                          "the chatbot returns 'query failed' for content questions!")
    
    def test_anthropic_api_key_present(self):
        """Test that API key is configured"""
        from config import config as real_config
        
        # Note: In tests this might be empty, but we should check it's at least defined
        self.assertTrue(hasattr(real_config, 'ANTHROPIC_API_KEY'), 
                       "ANTHROPIC_API_KEY should be defined in config")
    
    def test_chroma_path_configuration(self):
        """Test that ChromaDB path is configured"""
        from config import config as real_config
        
        self.assertEqual(real_config.CHROMA_PATH, "./chroma_db",
                        "ChromaDB path should be configured correctly")


if __name__ == '__main__':
    print("=" * 60)
    print("TESTING RAG System query handling")
    print("=" * 60)
    
    unittest.main(verbosity=2)