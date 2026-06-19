"""
Builds and loads the Chroma vectorstore using Ollama embeddings.
"""
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

from src import config


def get_embedding_model():
    return OllamaEmbeddings(model=config.EMBEDDING_MODEL)


def build_vectorstore(splits, persist_directory=None):
    embeddings = get_embedding_model()
    persist_directory = persist_directory or config.VECTORSTORE_DIR
    return Chroma.from_documents(
        documents=splits, embedding=embeddings, persist_directory=persist_directory
    )


def load_vectorstore(persist_directory=None):
    embeddings = get_embedding_model()
    persist_directory = persist_directory or config.VECTORSTORE_DIR
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)
