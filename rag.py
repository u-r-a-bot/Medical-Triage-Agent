# rag.py
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

VECTOR_STORE_PATH = "vector_store"
KNOWLEDGE_BASE_PATH = "knowledge_base"

def build_vector_store():
    """Builds and saves a FAISS vector store from documents in the knowledge base."""
    if os.path.exists(VECTOR_STORE_PATH):
        print("Vector store already exists. Skipping build.")
        return

    print("Building vector store...")
    # Load documents from the knowledge_base directory
    loader = DirectoryLoader(
        KNOWLEDGE_BASE_PATH,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True
    )
    documents = loader.load()

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    
    # Use a local embedding model
    print("Loading embedding model...")
    # Using a popular, efficient model for this task
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create the FAISS vector store and save it locally
    print("Creating and saving FAISS vector store...")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(VECTOR_STORE_PATH)
    print("Vector store built and saved.")

def get_retriever():
    """Loads the FAISS vector store and returns a retriever."""
    if not os.path.exists(VECTOR_STORE_PATH):
        print("Vector store not found. Building it now...")
        build_vector_store()
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # Return a retriever that finds the top 3 most relevant documents
    return db.as_retriever(search_kwargs={"k": 3})

if __name__ == '__main__':
    # This allows you to build the store by running `python rag.py`
    build_vector_store()