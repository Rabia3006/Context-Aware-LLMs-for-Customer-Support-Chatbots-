"""
Retriever implementations:
- vectorstore retriever: real semantic retrieval (the actual RAG retriever)
- random_retriever: sanity-check baseline that ignores the query entirely
"""
import random
from langchain.docstore.document import Document

from src import config


def get_vectorstore_retriever(vectorstore, k=None):
    k = k or config.TOP_K
    return vectorstore.as_retriever(search_kwargs={"k": k})


def random_retriever(vectorstore, k=1):
    """Returns k random chunks regardless of the query."""
    docs = vectorstore._collection.get(include=["documents", "metadatas"])
    all_docs = docs["documents"]
    all_metas = docs["metadatas"]
    indices = random.sample(range(len(all_docs)), k)
    return [
        Document(page_content=all_docs[i], metadata=all_metas[i] if all_metas[i] is not None else {})
        for i in indices
    ]
