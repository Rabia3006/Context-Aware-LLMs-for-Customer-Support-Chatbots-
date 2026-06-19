"""
LLM generation and the RAG chain that ties retrieval + generation together.
"""
import ollama
from src import config


def ollama_llm(question, context, model=None):
    model = model or config.GENERATOR_MODEL
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model=model, messages=[{"role": "user", "content": formatted_prompt}])
    return response["message"]["content"]


def rag_chain(question, retriever):
    retrieved_docs = retriever.invoke(question)
    formatted_context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return ollama_llm(question, formatted_context)
