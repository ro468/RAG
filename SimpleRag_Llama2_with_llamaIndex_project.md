# Attention is All You Need Q&A System

This markdown file provides a detailed explanation of the steps taken to create a Q&A system based on the "Attention is All You Need" paper. The system leverages various tools and libraries, including `HuggingFaceLLM`, `llama_index`, and `LangChain`.

---
## Prerequisites

First, ensure you have the necessary libraries installed. Here are the commands to install the required packages:

```bash
!pip install pypdf
!pip install -q transformers einops accelerate langchain bitsandbytes
!pip install sentence_transformers
!pip install llama_index
pip install llama-index-llms-huggingface
pip install -U llama-index-readers-file
pip install -U langchain-community
pip install llama-index-embeddings-langchain
```

## Step 1: Load and Process the Data

Download the "Attention is All You Need" paper in PDF format and place it in the `/content/data` directory. Then, use the `SimpleDirectoryReader` to load the data.

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
documents = SimpleDirectoryReader("/content/data").load_data()
print(documents)
```

## Step 2: Define Prompts

### System Prompt

The system prompt sets the role and behavior of the assistant. It provides context and instructions for how the assistant should respond to queries.

```python
system_prompt = """
You are a Q&A assistant. Your goal is to answer questions as
accurately as possible based on the instructions and context provided.
"""
```

### Query Wrapper Prompt

The query wrapper prompt formats the user queries in a way that the model can understand and process effectively.

```python
from llama_index.core import PromptTemplate
query_wrapper_prompt = PromptTemplate("{query_str}")
```

## Step 3: Authenticate with Hugging Face

Authenticate with Hugging Face to access the models.

```bash
!huggingface-cli login
```

## Step 4: Initialize the HuggingFaceLLM

Initialize the `HuggingFaceLLM` with the required parameters. This includes setting up the tokenizer and model with memory-efficient parameters.

```python
from llama_index.llms.huggingface import HuggingFaceLLM
import torch

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.7, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="StabilityAI/stablelm-tuned-alpha-3b",
    model_name="StabilityAI/stablelm-tuned-alpha-3b",
    device_map="auto",
    stopping_ids=[50278, 50279, 50277, 1, 0],
    tokenizer_kwargs={"max_length": 4096},
    model_kwargs={"torch_dtype": torch.float16}
)
```

### Explanation:
- **HuggingFaceLLM**: This class is used to initialize the language model from Hugging Face with specific settings such as context window, token generation parameters, prompts, and memory management configurations.
- **context_window**: The maximum number of tokens that the model can process in a single forward pass.
- **max_new_tokens**: The maximum number of new tokens to generate in response to a query.
- **generate_kwargs**: Additional arguments for text generation, such as temperature and sampling settings.
- **system_prompt and query_wrapper_prompt**: Prompts that guide the model's behavior and format the input query.
- **device_map and model_kwargs**: Settings to optimize model loading and execution, including device allocation and data types to reduce memory usage.

## Step 5: Configure Settings

Set the LLM and chunk size in the settings.

```python
from llama_index.core import Settings

Settings.llm = llm
Settings.chunk_size = 1024
```

### Explanation:
- **Settings**: Configuration parameters for the system, including the language model and the chunk size for splitting large documents into manageable pieces.
- **chunk_size**: The number of tokens in each chunk when processing long documents.

## Step 6: Set Up Embeddings

Use the `LangchainEmbedding` to set up the embedding model.

```python
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding

embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
)
```

### Explanation:
- **Embedding Model**: Transforms text into numerical representations (embeddings) that can be used for similarity search and other NLP tasks.
- **HuggingFaceEmbeddings**: A specific embedding model from Hugging Face's library.
- **LangchainEmbedding**: A wrapper that integrates the embedding model with `llama_index`.

## Step 7: Create the Service Context

Create the service context with the LLM and embedding model.

```python
service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embed_model
)
```

### Explanation:
- **ServiceContext**: A container for the configuration and resources needed by the system, including the LLM, embedding model, and chunk size. It ensures that all components work together seamlessly.

## Step 8: Build the Index

Build the vector store index from the documents and the service context.

```python
index = VectorStoreIndex.from_documents(documents, service_context=service_context)
print(index)
```

### Explanation:
- **VectorStoreIndex**: An index that stores document embeddings for efficient similarity search. It enables quick retrieval of relevant document segments based on the user's query.
- **from_documents**: A method to create the index from a list of documents using the provided service context.

## Step 9: Query the Index

Convert the index to a query engine and query it with a sample question.

```python
query_engine = index.as_query_engine()
response = query_engine.query("what is attention is all you need?")
print(response)
```

### Explanation:
- **Query Engine**: An interface to interact with the index and retrieve answers to user queries.
- **query**: A method to submit a query to the index and receive a response based on the indexed documents.

---
By following these steps, you can create a Q&A system based on the "Attention is All You Need" paper using state-of-the-art NLP models and tools. This system leverages advanced language models, efficient indexing, and embedding techniques to provide accurate and relevant answers to user queries.