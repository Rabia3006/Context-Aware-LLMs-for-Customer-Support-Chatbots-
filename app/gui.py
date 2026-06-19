"""
Interactive Panel GUI for querying the RAG pipeline.
Run with: panel serve app/gui.py
"""
import panel as pn
from typing import List
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from src import config
from src.vectorstore import load_vectorstore
from src.retrievers import get_vectorstore_retriever

pn.extension("bootstrap")

llm = Ollama(model=config.GENERATOR_MODEL)
vectorstore = load_vectorstore()
retriever = get_vectorstore_retriever(vectorstore)

prompt = PromptTemplate.from_template(
    "Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
)
rag_chain = RetrievalQA.from_chain_type(
    llm=llm, retriever=retriever, chain_type="stuff", chain_type_kwargs={"prompt": prompt}
)


def rag_chain_fn(question: str) -> str:
    return rag_chain.invoke({"query": question})["result"]


query_input = pn.widgets.TextAreaInput(name="🔎 Enter your question", placeholder="Type your question here…",
                                         height=120, max_length=5000, width=400)
k_input = pn.widgets.IntSlider(name="📄 Context documents (k)", start=1, end=10, value=2)
run_button = pn.widgets.Button(name="Run", button_type="primary")
clear_button = pn.widgets.Button(name="Clear", button_type="warning")
status_alert = pn.pane.Alert("✅ Ready.", alert_type="success", visible=True, width=600)
answer_md = pn.pane.Markdown("Answer will appear here.")
contexts_md = pn.pane.Markdown("Contexts will appear here.")


def _format_contexts(texts: List[str]) -> str:
    if not texts:
        return "_No contexts retrieved._"
    return "\n\n---\n\n".join(f"**Context {i}:**\n\n```\n{t}\n```" for i, t in enumerate(texts, 1))


def on_run_click(event):
    try:
        status_alert.alert_type, status_alert.object = "info", "⏳ Running…"
        q = (query_input.value or "").strip()
        if not q:
            status_alert.alert_type, status_alert.object = "warning", "⚠️ Please enter a question."
            return
        k = int(k_input.value)
        docs = retriever.invoke(q)[:k]
        contexts_md.object = _format_contexts([d.page_content for d in docs])
        answer_md.object = rag_chain_fn(q)
        status_alert.alert_type, status_alert.object = "success", "✅ Done."
    except Exception as e:
        status_alert.alert_type, status_alert.object = "danger", f"❌ Error: {e}"


def on_clear_click(event):
    query_input.value = ""
    answer_md.object = "Answer will appear here."
    contexts_md.object = "Contexts will appear here."
    status_alert.alert_type, status_alert.object = "success", "✅ Cleared."


run_button.on_click(on_run_click)
clear_button.on_click(on_clear_click)

template = pn.template.BootstrapTemplate(
    title="📘 RAG Customer Support Chatbot",
    sidebar=[pn.pane.Markdown("## Controls"), query_input, k_input, pn.Row(run_button, clear_button)],
    main=[status_alert,
          pn.Card(answer_md, title="💡 Answer", collapsible=False, width=600),
          pn.Card(contexts_md, title="📚 Retrieved Contexts", collapsible=True, width=600)],
)
template.servable()
