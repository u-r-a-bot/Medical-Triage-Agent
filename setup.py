#!/usr/bin/env python3
"""
Setup script for the Clinical Triage Agent.
This script automates the installation and setup process.
"""

import os
import sys
import subprocess
import shutil

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. Current version: {version.major}.{version.minor}")
        return False
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install required Python packages."""
    print("\n📦 Installing dependencies...")
    
    # Upgrade pip first
    if not run_command("pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        return False
    
    return True

def setup_environment():
    """Set up environment variables."""
    print("\n🔧 Setting up environment...")
    
    env_file = ".env"
    if not os.path.exists(env_file):
        print("📝 Creating .env file...")
        with open(env_file, "w") as f:
            f.write("# Clinical Triage Agent Environment Variables\n")
            f.write("# Add your Gemini API key below\n")
            f.write("GEMINI_API_KEY=your_gemini_api_key_here\n")
        print("✅ .env file created")
        print("⚠️  Please edit .env file and add your Gemini API key")
    else:
        print("✅ .env file already exists")
    
    return True

def build_knowledge_base():
    """Build the knowledge base from the dataset."""
    print("\n📚 Building knowledge base...")
    
    if not os.path.exists("Training.csv"):
        print("❌ Training.csv not found. Please download the dataset first.")
        print("   You can download it from: https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning")
        return False
    
    if not run_command("python prep_dataset.py", "Building knowledge base"):
        return False
    
    return True

def build_vector_store():
    """Build the vector store for RAG."""
    print("\n🔍 Building vector store...")
    
    if not run_command("python rag.py", "Building vector store"):
        return False
    
    return True

def main():
    """Main setup function."""
    print("🚀 Clinical Triage Agent - Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        print("❌ Setup failed due to incompatible Python version")
        return False
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed during dependency installation")
        return False
    
    # Setup environment
    if not setup_environment():
        print("❌ Setup failed during environment setup")
        return False
    
    # Build knowledge base
    if not build_knowledge_base():
        print("⚠️  Knowledge base setup skipped (dataset not found)")
        print("   You can run 'python prep_dataset.py' later when you have the dataset")
    
    # Build vector store
    if not build_vector_store():
        print("❌ Setup failed during vector store creation")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env file and add your Gemini API key")
    print("2. Run the test script: python test_setup.py")
    print("3. Start the application: streamlit run app.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 