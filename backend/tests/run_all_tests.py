#!/usr/bin/env python3
"""
Comprehensive test runner for RAG chatbot debugging
"""
import unittest
import sys
import os
from io import StringIO

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import test modules
from test_course_search_tool import TestCourseSearchTool
from test_ai_generator import TestAIGenerator, TestAIGeneratorIntegration
from test_rag_system import TestRAGSystem, TestRAGSystemIntegration, TestRAGSystemRealConfigIssues


class RAGSystemTestRunner:
    """Custom test runner to analyze RAG system issues"""
    
    def __init__(self):
        self.issues_found = []
        self.passed_tests = []
        self.failed_tests = []
    
    def run_all_tests(self):
        """Run all tests and collect results"""
        print("=" * 80)
        print("RAG CHATBOT DEBUGGING - COMPREHENSIVE TEST ANALYSIS")  
        print("=" * 80)
        print()
        
        # Test suites in order of system layers
        test_suites = [
            ("1. CourseSearchTool Tests", [TestCourseSearchTool]),
            ("2. AIGenerator Tests", [TestAIGenerator, TestAIGeneratorIntegration]),
            ("3. RAG System Tests", [TestRAGSystem, TestRAGSystemIntegration, TestRAGSystemRealConfigIssues])
        ]
        
        for suite_name, test_classes in test_suites:
            print(f"\n{suite_name}")
            print("-" * len(suite_name))
            
            suite = unittest.TestSuite()
            for test_class in test_classes:
                tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
                suite.addTest(tests)
            
            # Capture test output
            stream = StringIO()
            runner = unittest.TextTestRunner(stream=stream, verbosity=2)
            result = runner.run(suite)
            
            # Analyze results
            self._analyze_test_results(suite_name, result, stream.getvalue())
        
        # Generate comprehensive report
        self._generate_report()
    
    def _analyze_test_results(self, suite_name, result, output):
        """Analyze test results and categorize issues"""
        print(f"Tests run: {result.testsRun}, Failures: {len(result.failures)}, Errors: {len(result.errors)}")
        
        # Track passed tests
        if result.testsRun > 0:
            passed = result.testsRun - len(result.failures) - len(result.errors)
            self.passed_tests.append((suite_name, passed, result.testsRun))
        
        # Analyze failures
        for test, traceback in result.failures:
            test_name = str(test)
            
            # Categorize specific issues
            if "MAX_RESULTS" in traceback and "0 not greater than 0" in traceback:
                self.issues_found.append({
                    'type': 'CRITICAL_CONFIG_ISSUE',
                    'component': 'Config',
                    'issue': 'MAX_RESULTS is set to 0',
                    'impact': 'Vector searches return 0 results, causing query failures',
                    'fix': 'Change MAX_RESULTS from 0 to 5 in config.py line 24',
                    'severity': 'HIGH'
                })
            
            elif "specific topic" in traceback:
                self.issues_found.append({
                    'type': 'TEST_ISSUE',
                    'component': 'CourseSearchTool Test',
                    'issue': 'Case sensitive string matching in test',
                    'impact': 'Test validation error, not system issue',
                    'fix': 'Update test to match actual output format',
                    'severity': 'LOW'
                })
            
            elif "execute_calls" in traceback and "0 != 1" in traceback:
                self.issues_found.append({
                    'type': 'TEST_MOCK_ISSUE',
                    'component': 'AIGenerator Test',
                    'issue': 'Mock tool execution not being called as expected',
                    'impact': 'Test design issue, may indicate real integration problem',
                    'fix': 'Fix mock setup or verify actual tool execution flow',
                    'severity': 'MEDIUM'
                })
            
            self.failed_tests.append((test_name, traceback))
        
        # Analyze errors
        for test, traceback in result.errors:
            self.failed_tests.append((str(test), traceback))
    
    def _generate_report(self):
        """Generate comprehensive diagnosis report"""
        print("\n" + "=" * 80)
        print("DIAGNOSIS REPORT: RAG CHATBOT 'QUERY FAILED' ISSUE")
        print("=" * 80)
        
        print(f"\n📊 TEST SUMMARY:")
        print(f"{'='*50}")
        for suite_name, passed, total in self.passed_tests:
            print(f"{suite_name}: {passed}/{total} tests passed")
        
        if not self.issues_found:
            print("\n✅ No major issues found in tested components")
            return
        
        print(f"\n🔍 ISSUES IDENTIFIED ({len(self.issues_found)}):")
        print("="*50)
        
        # Group issues by severity
        critical = [i for i in self.issues_found if i['severity'] == 'HIGH']
        medium = [i for i in self.issues_found if i['severity'] == 'MEDIUM']
        low = [i for i in self.issues_found if i['severity'] == 'LOW']
        
        # Critical issues first
        if critical:
            print("\n🚨 CRITICAL ISSUES (Fix these first!):")
            for i, issue in enumerate(critical, 1):
                print(f"\n{i}. Component: {issue['component']}")
                print(f"   Issue: {issue['issue']}")
                print(f"   Impact: {issue['impact']}")
                print(f"   Fix: {issue['fix']}")
        
        if medium:
            print("\n⚠️  MEDIUM PRIORITY ISSUES:")
            for i, issue in enumerate(medium, 1):
                print(f"\n{i}. Component: {issue['component']}")
                print(f"   Issue: {issue['issue']}")
                print(f"   Fix: {issue['fix']}")
        
        if low:
            print("\n📝 LOW PRIORITY ISSUES (Test-related):")
            for i, issue in enumerate(low, 1):
                print(f"\n{i}. Component: {issue['component']}")
                print(f"   Issue: {issue['issue']}")
                print(f"   Fix: {issue['fix']}")
        
        print(f"\n🔧 RECOMMENDED FIXES:")
        print("="*50)
        print("\n1. IMMEDIATE FIX (Likely solves the main problem):")
        print("   Edit backend/config.py line 24:")
        print("   Change: MAX_RESULTS: int = 0")  
        print("   To:     MAX_RESULTS: int = 5")
        print("\n   This is likely the root cause of 'query failed' errors!")
        
        print("\n2. VERIFY AFTER FIX:")
        print("   - Restart the application")
        print("   - Test a content-related query")
        print("   - Check if search results are returned")
        
        print("\n3. ADDITIONAL CHECKS:")
        print("   - Ensure ANTHROPIC_API_KEY is set in .env file")
        print("   - Verify ChromaDB database has course content") 
        print("   - Check that course documents are in ./docs/ folder")
        
        print(f"\n📋 NEXT STEPS:")
        print("="*50)
        print("1. Apply the MAX_RESULTS fix")
        print("2. Re-run tests to verify fix")
        print("3. Test the chatbot with actual queries")
        print("4. Monitor for any remaining issues")


if __name__ == '__main__':
    runner = RAGSystemTestRunner()
    runner.run_all_tests()