from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma, FAISS, ScaNN, Annoy
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import SystemMessage, HumanMessage, AIMessage

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

# Function to create RAG chain
def create_rag_chain(llm, vector_database_directory, method='Chroma', embedding_function=None, system_prompt='', chat_history_prompt=''):
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
    elif method == 'ScaNN':
        vectorstore = ScaNN(
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
    
    # Define LLM with user-provided parameters
    llm = llm

    # * Combine chat history with RAG retriever
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", chat_history_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # * 2. Answer question based on Chat Context
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
    
    return RunnableWithMessageHistory(
        rag_chain, 
        lambda session_id: msgs,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
