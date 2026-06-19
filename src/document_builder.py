"""
Converts raw dataset rows into LangChain Documents and chunks them.
"""
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src import config


def build_documents(train_dataset):
    """Turn each (instruction, response) pair into a single Document."""
    docs = []
    for item in train_dataset:
        text = f"Customer: {item['instruction']}\nSupport: {item['response']}"
        docs.append(Document(page_content=text))
    return docs


def chunk_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    return text_splitter.split_documents(docs)
