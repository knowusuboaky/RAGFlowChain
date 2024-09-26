# RAGFlowChain

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/ragflowchain.svg)](https://pypi.org/project/ragflowchain/)

**RAGFlowChain** is a powerful and flexible toolkit designed for building Retrieval-Augmented Generation (RAG) pipelines. This library integrates data loading from various sources, vector database creation, and chain management, making it easier to develop advanced AI solutions that combine retrieval mechanisms with generative models.

## Features

- **Comprehensive Data Loading**: Fetch and process data from books, news articles, YouTube videos, websites, and local documents.
- **Vector Database Creation**: Create and manage vector databases using state-of-the-art embeddings with Chroma vector stores.
- **RAG Chain Management**: Seamlessly integrate retrieval with language models to build context-aware AI systems.
- **Extensible Design**: Easily extend and customize the pipeline to fit specific use cases.

## Workflow

<img src="https://github.com/knowusuboaky/RAGFlowChain/blob/main/README_files/figure-markdown/mermaid-figure-1.png?raw=true" width="2000" height="450" alt="Optional Alt Text">

## Installation

To install RAGFlowChain, simply run:

```bash
pip install RAGFlowChain
```

## Quickstart

### 1. Fetch Data from Multiple Sources

RAGFlowChain allows you to fetch and process data from various online and local sources, all integrated into a single DataFrame.

```python
from ragflowchain import data_loader

# Define online and local data sources
online_sources = {
    'books': ("YOUR_BOOKS_API_KEY", 'automobile industry', 10),
    'news_articles': ("YOUR_NEWS_API_KEY", 'automobile marketing', 5, 1),
    'youtube': ('honda acura', "YOUR_YOUTUBE_API_KEY", 10),
    'websites': ["https://www.example.com", "https://www.anotherexample.com"]
}

local_sources = ["../folder/irt.csv", "../book/book.pdf", "../documents/sample.docx", "../notes/note.txt"]

# Fetch and process the data
final_data_df = data_loader(online_sources=online_sources, local_sources=local_sources, chunk_size=1000)

# Display the DataFrame
print(final_data_df)
```

#### Return Values:

- **`final_data_df`**: 
  - **Type**: `pandas.DataFrame`
  - **Description**: The final DataFrame containing the processed data from all specified sources. It includes columns such as `source`, `title`, `author`, `publishedDate`, `description`, `content`, `url`, and `source_type`. Each row corresponds to a chunk of content from a source, making it ready for further processing or embedding.

### 2. Create a Vector Database

Once you have the data, you can create a vector database using the `create_database` function.

```python
from ragflowchain import create_database

# Create a vector store from the processed data
vectorstore, docs_recursive = create_database(
    df=final_data_df,
    page_content="content",
    embedding_function=None,  # Uses default SentenceTransformerEmbeddings
    vectorstore_directory="data/chroma.db",
    chunk_size=1000,
    chunk_overlap=100
)
```

#### Explanation of `create_database` Arguments:

- **`df`**: 
  - **Type**: `pandas.DataFrame`
  - **Description**: This is the DataFrame containing the processed data from various sources. The DataFrame should include a column with the main text content (`page_content`) that you want to split into chunks and store in the vector database. Other columns might include metadata like `source`, `title`, `author`, etc.

- **`page_content`**: 
  - **Type**: `str`
  - **Description**: The name of the column in the DataFrame (`df`) that contains the main text content. This content will be split into chunks and used to create the embeddings that are stored in the vector database.

- **`embedding_function`**: 
  - **Type**: (Optional) A function or model
  - **Description**: A function or pre-trained model used to generate embeddings for the text chunks. If not provided, it defaults to using the `SentenceTransformerEmbeddings` model from `sentence-transformers`, specifically the "all-MiniLM-L6-v2" model. This model converts text chunks into high-dimensional vectors that can be stored in the vector database.

- **`vectorstore_directory`**: 
  - **Type**: `str`
  - **Description**: The file path to the directory where the vector store will be saved. This directory will store the Chroma vector database, allowing you to persist and reload the database across different sessions. The default path is `"data/chroma.db"`.

- **`chunk_size`**: 
  - **Type**: `int`
  - **Description**: The maximum number of characters in each text chunk after splitting. This parameter is crucial for ensuring that the chunks are small enough to be processed efficiently but large enough to maintain context. The default value is `1000` characters.

- **`chunk_overlap`**: 
  - **Type**: `int`
  - **Description**: The number of overlapping characters between consecutive chunks. Overlap helps to maintain context across chunks, especially when important information might be split between two chunks. The default overlap is `100` characters.

#### Return Values:

- **`vectorstore`**: 
  - **Type**: `Chroma`
  - **Description**: The Chroma vector store object containing the embeddings for the text chunks. This vector store is saved in the specified directory (`vectorstore_directory`) and can be used for retrieval tasks.

- **`docs_recursive`**: 
  - **Type**: `List[Document]`
  - **Description**: A list of document chunks after recursive splitting. Each chunk is an instance of the `Document` class, containing both the content and metadata such as source, title, and other relevant information from the original DataFrame.

### 3. Build a RAG Chain

Integrate the data and vector store into a Retrieval-Augmented Generation (RAG) chain.

```python
from ragflowchain import create_rag_chain

rag_chain = create_rag_chain(
    llm=YourLanguageModel(),  # Replace with your LLM instance
    vector_database_directory="data/chroma.db",
    embedding_function=None,  # Optional, defaults to SentenceTransformerEmbeddings
    system_prompt="This is a system prompt.",
    chat_history_prompt="This is a chat history prompt."
)
```

#### Explanation of `create_rag_chain` Arguments:

- **`llm`**: 
  - **Type**: Language model instance
  - **Description**: The language model that will be used in the RAG chain. This could be an instance of GPT-3 or any other compatible model.

- **`vector_database_directory`**: 
  - **Type**: `str`
  - **Description**: The directory where the vector store is located. The vector store contains the embeddings generated from the data.

- **`embedding_function`**: 
  - **Type**: (Optional) A function or model
  - **Description**: The function or model used to generate embeddings during retrieval. Defaults to `SentenceTransformerEmbeddings` if not provided.

- **`system_prompt`**: 
  - **Type**: `str`
  - **Description**: A prompt given to the language model to guide its responses. This could include instructions or context specific to the application.

- **`chat_history_prompt`**: 
  - **Type**: `str`
  - **Description**: A prompt template that incorporates the chat history, helping the model maintain context across multiple interactions.

#### Return Values:

- **`rag_chain`**: 
  - **Type**: `RunnableWithMessageHistory`
  - **Description**: The RAG chain object that can be used to process user inputs and generate context-aware responses by combining retrieval and generative capabilities.

### 4. Run the RAG Chain Using `invoke`

```python
# Example usage with invoke method
result = rag_chain.invoke(
    {"input": "Your question here"}, 
    config={
        "configurable": {"session_id": "user123"}
    }
)

print(result["answer"])
```

#### Explanation of `invoke` Usage:

- **`invoke`**: This method is used to trigger the execution of the RAG chain. It’s preferred over `run` when working with configurable settings or when `invoke` is the designated method in the LangChain framework you’re using.

- **Input Dictionary**: The user’s question or input is passed as a dictionary with the key `"input"`.

- **`config` Dictionary**: Additional configurations can be passed using the `config` dictionary. Here, `"configurable": {"session_id": "user123"}` sets a session ID, which is useful for tracking the conversation history across multiple interactions.

#### Return Values:

- **`result`**: 
  - **Type**: `dict`
  - **Description**: A dictionary containing the model's response. The key `"answer"` contains the generated response to the user's input, while other keys might include additional metadata depending on the RAG chain configuration.

## Detailed Explanation of Function Arguments

### 1. `data_loader`

```python
data_loader(online_sources=None, local_sources=None, chunk_size=1000)
```

- **`online_sources`**: A dictionary specifying the online sources from which to fetch data. The keys represent the type of source, and the values are tuples containing the necessary parameters.
  - **`books`**: A tuple (`api_key`, `query`, `max_results`) for fetching books from Google Books API.
  - **`news_articles`**: A tuple (`api_key`, `query`, `page_size`, `max_pages`) for fetching news articles from NewsAPI.
  - **`youtube`**: A tuple (`query`, `api_key`, `max_results`) for fetching YouTube videos.
  - **`websites`**: A list of URLs to fetch content from websites.

- **`local_sources`**: A list of paths to local files (PDF, CSV, DOCX, TXT). The function loads and processes these documents into manageable chunks.

- **`chunk_size`**: The size of each text chunk, in characters. Default is `1000`. This determines how the text content is split, ensuring that chunks are neither too large nor too small.

### 2. `create_database`

```python
create_database(df, page_content, embedding_function=None, vectorstore_directory="data/chroma.db", chunk_size=1000, chunk_overlap=100)
```

- **`df`**: A pandas DataFrame containing the processed data. This should include columns like `content`, `source`, etc.

- **`page_content`**: The name of the column in the DataFrame that contains the main text content to be embedded.

- **`embedding_function`**: (Optional) A function or model used to generate embeddings. Defaults to using `SentenceTransformerEmbeddings`.

- **`vectorstore_directory`**: The directory where the vector store will be saved. Default is `"data/chroma.db"`.

- **`chunk_size`**: The size of each text chunk, in characters. Used during text splitting.

- **`chunk_overlap`**: The overlap between consecutive chunks, to maintain context. Default is `100`.

### 3. `create_rag_chain`

```python
create_rag_chain(llm, vector_database_directory, embedding_function=None, system_prompt='', chat_history_prompt='')
```

- **`llm`**: The language model that will be used in the RAG chain. This could be an instance of GPT-3 or any other compatible model.

- **`vector_databse_directory`**: The directory where the vector database is located.

- **`embedding_function`**: (Optional) The function or model used to generate embeddings during retrieval.

- **`system_prompt`**: A prompt given to the language model to guide its responses. This could include instructions or context.

- **`chat_history_prompt`**: A prompt template that incorporates the chat history, helping the model maintain context across multiple interactions.

### 4. `rag_chain.invoke`

```python
rag_chain.invoke({"input": question}, config={"configurable": {"session_id": "any"}})
```

- **`input`**: The user’s input or question, passed as a dictionary with the key `"input"`. This is the prompt or query that the model will process.

- **`config`**: A dictionary containing additional configuration settings. The `"configurable"` key allows you to set a session ID or other parameters that influence how the chain processes the input.

## Documentation

For more detailed documentation, including advanced usage and customization options, please visit the [GitHub repository](https://github.com/knowusuboaky/RAGFlowChain).

## License

RAGFlowChain is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Acknowledgements

RAGFlowChain is built on top of powerful tools like [LangChain](https://github.com/hwchase17/langchain) and [Chroma](https://github.com/chroma-core/chroma). We thank the open-source community for their contributions.

Made with ❤️ by Kwadwo Daddy Nyame Owusu - Boakye. [[Email]](kwadwo.owusuboakye@outlook.com)

