# RAGFlowChain/__init__.py

# Import the main fetch_data function from data_fetcher.py
from .data_fetcher import fetch_data as data_loader
from .create_vectorstore import create_vectorstore as create_database
from .create_rag_chain import create_rag_chain as create_rag_chain

# Define what is available to import directly from raggrippa
__all__ = ["data_loader", "create_database", "create_rag_chain"]
