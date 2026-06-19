"""
Builds the document index (vectorstore) from the Bitext dataset.
Run once before evaluation or the app: python scripts/build_index.py
"""
import os
from src import config
from src.data_loader import load_bitext_dataset, split_train_test
from src.document_builder import build_documents, chunk_documents
from src.vectorstore import build_vectorstore
from datasets import Dataset


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print("Loading dataset...")
    df = load_bitext_dataset()
    train_df, test_df = split_train_test(df)
    train_dataset = Dataset.from_pandas(train_df)

    print("Building and chunking documents...")
    docs = build_documents(train_dataset)
    splits = chunk_documents(docs)
    print(f"{len(splits)} chunks created.")

    print("Embedding and building vectorstore (calls Ollama, may take a while)...")
    build_vectorstore(splits)
    print(f"Vectorstore persisted to {config.VECTORSTORE_DIR}")

    test_df.to_csv(f"{config.RESULTS_DIR}/test_split.csv", index=False)
    print("Test split saved for evaluation step.")


if __name__ == "__main__":
    main()
