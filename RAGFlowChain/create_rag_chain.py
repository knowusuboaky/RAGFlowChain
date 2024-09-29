from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma, FAISS, Annoy
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from langchain_community.tools import TavilySearchResults
import time

# Simple wrapper class for message history
class SimpleMessageHistory:
    def __init__(self):
        self.messages = []

    def add_messages(self, new_messages):
        """Add new messages to the history."""
        self.messages.extend(new_messages)

    def get_messages(self):
        """Retrieve the full message history."""
        return self.messages

    def clear_messages(self):
        """Clear all messages from the history."""
        self.messages = []

# Default system prompt for question-answering tasks
default_system_prompt = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If there is any real-time weather information available, prioritize including that in your response. 
If you don't know the answer, or the information is not available, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

{context}"""

# Default prompt for contextualizing questions
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

# Function to perform web search and combine with vectorstore documents
def perform_web_search(input_text, vectorstore):
    # Initialize web search tool
    web_search_tool = TavilySearchResults(max_results=5)

    # Perform web search
    docs = web_search_tool.invoke({"query": input_text})

    # Introduce a slight delay after the web search
    time.sleep(2)

    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    
    # Retrieve documents from vectorstore
    documents = vectorstore.retrieve(input_text)

    # Combine web results with vectorstore documents
    if documents is None and web_results is not None:
        documents = [web_results]
    elif documents is not None and web_results is not None:
        documents.append(web_results)
    elif documents is not None and web_results is None:
        # If there are documents from the vector store but no web results,
        # simply return the documents retrieved from the vector store.
        pass
    
    # Return the combined documents
    return documents

# Function to create RAG chain with web search integration
def create_rag_chain(
    llm, 
    vector_database_directory, 
    method='Chroma', 
    embedding_function=None, 
    system_prompt=None,  # Set to None by default to allow conditional assignment
    chat_history_prompt=None,  # Set to None by default to allow conditional assignment
    tavily_search=None  # Added argument for TavilySearchResults or API key
):
    # 3.0 Text Embeddings
    if embedding_function is None:
        embedding_function = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
    
    # Create vectorstore based on the specified method
    if method == 'Chroma':
        vectorstore = Chroma(
            persist_directory=vector_database_directory,
            embedding_function=embedding_function
        )
    elif method == 'FAISS':
        vectorstore = FAISS(
            persist_directory=vector_database_directory,
            embedding_function=embedding_function
        )
    elif method == 'Annoy':
        vectorstore = Annoy(
            persist_directory=vector_database_directory,
            embedding_function=embedding_function
        )
    else:
        raise ValueError(f"Unsupported method: {method}. Choose from 'Chroma', 'FAISS', 'ScaNN', 'Annoy'.")

    retriever = vectorstore.as_retriever()
    
    # Initialize the TavilySearchResults tool if an API key or instance is provided
    if tavily_search is not None:
        if isinstance(tavily_search, TavilySearchResults):
            web_search_tool = tavily_search
        else:
            web_search_tool = TavilySearchResults(api_key=tavily_search)
    else:
        web_search_tool = None
    
    # Function to perform web search and combine with vectorstore documents
    def perform_web_search(input_text, vectorstore):
        if web_search_tool is not None:
            # Perform web search
            docs = web_search_tool.invoke({"query": input_text})

            # Introduce a slight delay after the web search
            time.sleep(2)

            web_results = "\n".join([d["content"] for d in docs])
            web_results = Document(page_content=web_results)
        else:
            web_results = None
        
        # Retrieve documents from vectorstore
        documents = vectorstore.retrieve(input_text)

        # Combine web results with vectorstore documents
        if documents is None and web_results is not None:
            documents = [web_results]
        elif documents is not None and web_results is not None:
            documents.append(web_results)
        elif documents is not None and web_results is None:
            # If there are documents from the vector store but no web results,
            # simply return the documents retrieved from the vector store.
            pass
        
        # Return the combined documents
        return documents

    # Use the provided system_prompt if not None, otherwise use the default prompt
    if system_prompt is None:
        system_prompt = default_system_prompt

    # Use the provided chat_history_prompt if not None, otherwise use the default contextualizing prompt
    if chat_history_prompt is None:
        chat_history_prompt = contextualize_q_system_prompt

    # * Combine chat history with RAG retriever
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", chat_history_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # * Answer question based on Chat Context
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # * Combine both RAG + Chat Message History
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    # Use SimpleMessageHistory here
    msgs = SimpleMessageHistory()
    
    # Create a custom function to invoke the RAG chain with web search and message history
    def custom_invoke(input_text):
        # Perform web search first
        documents = perform_web_search(input_text, vectorstore)
        
        # Return the required input structure
        return {
            "chat_history": msgs.get_messages(),
            "input": input_text,
            "documents": documents,
        }

    # Return the RunnableWithMessageHistory with the correct parameters
    return RunnableWithMessageHistory(
        get_session_history=lambda session_id: msgs,
        runnable=rag_chain,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
        custom_invoke_function=custom_invoke
    )
