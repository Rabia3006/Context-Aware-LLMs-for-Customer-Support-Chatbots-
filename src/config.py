"""
Central configuration for the RAG pipeline.
Reads sensitive values from environment variables (via .env) instead of
hardcoding them in the code.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Model configuration ---
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "llama3")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama3")

# --- API keys (optional) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# --- Dataset configuration ---
DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
TEST_SIZE = 0.2
EVAL_SAMPLE_SIZE = 100

# --- Chunking configuration ---
CHUNK_SIZE = 1059
CHUNK_OVERLAP = 211

# --- Retrieval configuration ---
TOP_K = 5

# --- Paths ---
VECTORSTORE_DIR = "data/chroma_store"
RESULTS_DIR = "results"
