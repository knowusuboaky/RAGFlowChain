from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import Chroma, FAISS, Annoy, ScaNN
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
import pandas as pd

def filter_complex_metadata(metadata):
    """
    This function filters out complex metadata values that are not str, int, float, or bool.
    If a value is None, it is replaced with an empty string to avoid errors.
    """
    filtered_metadata = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)):
            filtered_metadata[k] = v
        elif v is None:
            filtered_metadata[k] = ""  # Replace None with an empty string
    return filtered_metadata

def create_vectorstore(df, page_content, embedding_function=None, vectorstore_method='Chroma', vectorstore_directory="data/vectorstore.db", chunk_size=1000, chunk_overlap=100):
    """
    This function processes a DataFrame, including text preprocessing, splits the content into chunks, and stores the results in a vector database.

    Parameters:
    - df: pandas DataFrame containing the text data.
    - page_content: The column in the DataFrame that contains the text content.
    - embedding_function: The function used to generate embeddings. Defaults to SentenceTransformerEmbeddings.
    - vectorstore_method: The method to use for the vector store ('Chroma', 'FAISS', 'Annoy', 'ScaNN'). Defaults to 'Chroma'.
    - vectorstore_directory: Directory where the vectorstore will be saved. Defaults to "data/vectorstore.db".
    - chunk_size: The size of each text chunk. Defaults to 1000 characters.
    - chunk_overlap: The overlap size between chunks. Defaults to 100 characters.

    Returns:
    - vectorstore: The created vector store.
    - docs_recursive: The list of document chunks after recursive splitting.
    """
    
    # Text Preprocessing: Replace double newlines with a single newline
    df[page_content] = df[page_content].str.replace('\n\n', '\n', regex=False)
    
    # 1.0 Document Loaders
    loader = DataFrameLoader(df, page_content_column=page_content)
    documents = loader.load()
    
    # 2.0 Text Splitting (Recursive Character Splitter)
    text_splitter_recursive = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    docs_recursive = text_splitter_recursive.split_documents(documents)
    
    # Filter out complex metadata values
    for doc in docs_recursive:
        doc.metadata = filter_complex_metadata(doc.metadata)
    
    # 3.0 Text Embeddings
    if embedding_function is None:
        embedding_function = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
    
    # 4.0 Vector Store Creation
    if vectorstore_method == 'Chroma':
        vectorstore = Chroma.from_documents(
            docs_recursive, 
            embedding=embedding_function, 
            persist_directory=vectorstore_directory
        )
    elif vectorstore_method == 'FAISS':
        vectorstore = FAISS.from_documents(
            docs_recursive,
            embedding=embedding_function
        )
        vectorstore.save(vectorstore_directory)  # Save the FAISS index to the specified directory
    elif vectorstore_method == 'Annoy':
        vectorstore = Annoy.from_documents(
            docs_recursive,
            embedding=embedding_function,
            index_path=vectorstore_directory
        )
    elif vectorstore_method == 'ScaNN':
        vectorstore = ScaNN.from_documents(
            docs_recursive,
            embedding=embedding_function,
            persist_directory=vectorstore_directory
        )
    else:
        raise ValueError(f"Unsupported vectorstore method: {vectorstore_method}")
    
    return vectorstore, docs_recursive
