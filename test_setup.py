#!/usr/bin/env python3
"""
Test script to verify the clinical triage agent setup.
Run this script to check if all components are working correctly.
"""

import os
import sys
from dotenv import load_dotenv

def test_imports():
    """Test if all required packages can be imported."""
    print("🔍 Testing imports...")
    
    try:
        import streamlit
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import langchain
        print("✅ LangChain imported successfully")
    except ImportError as e:
        print(f"❌ LangChain import failed: {e}")
        return False
    
    try:
        import langgraph
        print("✅ LangGraph imported successfully")
    except ImportError as e:
        print(f"❌ LangGraph import failed: {e}")
        return False
    
    try:
        import faiss
        print("✅ FAISS imported successfully")
    except ImportError as e:
        print(f"❌ FAISS import failed: {e}")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ Sentence Transformers imported successfully")
    except ImportError as e:
        print(f"❌ Sentence Transformers import failed: {e}")
        return False
    
    try:
        import google.generativeai
        print("✅ Google Generative AI imported successfully")
    except ImportError as e:
        print(f"❌ Google Generative AI import failed: {e}")
        return False
    
    return True

def test_environment():
    """Test environment variables."""
    print("\n🔧 Testing environment...")
    
    load_dotenv()
    
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        print("✅ Gemini API key found")
        return True
    else:
        print("❌ Gemini API key not found. Please set GEMINI_API_KEY in your .env file")
        return False

def test_knowledge_base():
    """Test if knowledge base exists."""
    print("\n📚 Testing knowledge base...")
    
    knowledge_base_path = "knowledge_base"
    if os.path.exists(knowledge_base_path):
        files = os.listdir(knowledge_base_path)
        if len(files) > 0:
            print(f"✅ Knowledge base found with {len(files)} files")
            return True
        else:
            print("❌ Knowledge base directory is empty")
            return False
    else:
        print("❌ Knowledge base directory not found")
        return False

def test_vector_store():
    """Test if vector store exists."""
    print("\n🔍 Testing vector store...")
    
    vector_store_path = "vector_store"
    if os.path.exists(vector_store_path):
        files = os.listdir(vector_store_path)
        if "index.faiss" in files and "index.pkl" in files:
            print("✅ Vector store found")
            return True
        else:
            print("❌ Vector store files missing")
            return False
    else:
        print("❌ Vector store directory not found")
        return False

def test_rag_functionality():
    """Test RAG functionality."""
    print("\n🤖 Testing RAG functionality...")
    
    try:
        from rag import get_retriever
        retriever = get_retriever()
        print("✅ RAG retriever created successfully")
        
        # Test a simple query
        docs = retriever.get_relevant_documents("fever headache")
        if docs:
            print(f"✅ RAG retrieval working - found {len(docs)} documents")
            return True
        else:
            print("❌ RAG retrieval returned no documents")
            return False
            
    except Exception as e:
        print(f"❌ RAG functionality test failed: {e}")
        return False

def test_triage_agent():
    """Test triage agent functionality."""
    print("\n🏥 Testing triage agent...")
    
    try:
        from triage_agent import process_triage_request
        
        # Test with a simple case
        test_case = "I have a mild headache"
        result = process_triage_request(test_case)
        
        if result.get("success"):
            print("✅ Triage agent working correctly")
            print(f"   - Symptoms: {result.get('symptoms', [])}")
            print(f"   - Severity: {result.get('severity', '')}")
            print(f"   - Recommendation: {result.get('recommended_action', '')}")
            return True
        else:
            print(f"❌ Triage agent failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Triage agent test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Clinical Triage Agent - Setup Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_environment,
        test_knowledge_base,
        test_vector_store,
        test_rag_functionality,
        test_triage_agent
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your clinical triage agent is ready to use.")
        print("\nTo start the application, run:")
        print("   streamlit run app.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above and fix them.")
        print("\nCommon fixes:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Set your Gemini API key in .env file")
        print("3. Build the vector store: python rag.py")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 