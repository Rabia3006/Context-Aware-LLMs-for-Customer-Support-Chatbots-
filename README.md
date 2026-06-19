# Context-Aware LLMs for Customer Support Chatbots

A Retrieval-Augmented Generation (RAG) pipeline for answering customer support
questions, built on a local LLM (Ollama / llama3) and evaluated with RAGAS.

Originally built as a university dissertation project; now being extended and
restructured to explore additional RAG techniques (hybrid retrieval,
reranking, embedding fine-tuning).

## Project Structure
.

├── app/             # Interactive Panel GUI for querying the RAG pipeline

├── notebooks/        # Original exploratory notebook (Colab)

├── scripts/          # CLI entry points

│   ├── build_index.py     # builds the vectorstore from the Bitext dataset

│   └── run_evaluation.py  # runs RAGAS evaluation (random vs vectorstore retriever)

├── src/               # Core pipeline code

│   ├── config.py          # central settings, reads secrets from .env

│   ├── data_loader.py      # loads/splits the Bitext dataset

│   ├── document_builder.py # builds + chunks LangChain documents

│   ├── vectorstore.py       # Chroma vectorstore + Ollama embeddings

│   ├── retrievers.py         # vectorstore retriever + random baseline retriever

│   ├── generation.py          # Ollama generation + RAG chain

│   ├── evaluation.py           # RAGAS evaluation logic

│   └── visualization.py        # radar/bar charts comparing retrievers

├── requirements.txt

└── .env.example

## Setup

1. Install [Ollama](https://ollama.com) and pull the required models:
```bash
   ollama pull llama3
   ollama pull nomic-embed-text
```
2. Create a Python virtual environment and install dependencies:
```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
```
3. Copy `.env.example` to `.env` and fill in any values you need (Hugging Face
   token, OpenAI key if you plan to use an independent judge model):
```bash
   cp .env.example .env
```

## Usage

Build the vectorstore index from the Bitext dataset:
```bash
python scripts/build_index.py
```

Run the RAGAS evaluation (random retriever vs. real vectorstore retriever):
```bash
python scripts/run_evaluation.py
```

Launch the interactive GUI:
```bash
panel serve app/gui.py
```

## Known limitations / next steps

- The evaluation judge model is currently the same model used to generate
  answers (llama3), which can introduce self-evaluation bias. Swapping in an
  independent judge (e.g. GPT-4o-mini) would make the comparison more rigorous.
- The evaluation sample size is small (20 questions); scaling this up would
  give more reliable metrics.
- Planned additions: hybrid (BM25 + vector) retrieval, cross-encoder
  reranking, and fine-tuned embeddings, to compare against the baseline RAG
  pipeline.
