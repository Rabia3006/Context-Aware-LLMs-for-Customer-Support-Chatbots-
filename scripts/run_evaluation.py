"""
Evaluates the random-retriever baseline against the real vectorstore
retriever using RAGAS metrics.
Run after build_index.py: python scripts/run_evaluation.py
"""
import os
import pandas as pd

from src import config
from src.vectorstore import load_vectorstore
from src.retrievers import get_vectorstore_retriever, random_retriever
from src.evaluation import run_condition, print_averages
from src.visualization import plot_radar, plot_bar
from src.data_loader import sample_eval_set


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    test_df = pd.read_csv(f"{config.RESULTS_DIR}/test_split.csv")
    test_dataset = sample_eval_set(test_df)
    eval_df = pd.DataFrame(test_dataset)
    questions = eval_df["instruction"].tolist()
    ground_truths = eval_df["response"].tolist()

    vectorstore = load_vectorstore()
    retriever = get_vectorstore_retriever(vectorstore)

    random_df = run_condition(
        questions, ground_truths, lambda q: random_retriever(vectorstore), label="Random Retriever"
    )
    random_df.to_csv(f"{config.RESULTS_DIR}/random_retriever_results.csv", index=False)
    print_averages(random_df, "Random Retriever")

    vector_df = run_condition(
        questions, ground_truths, lambda q: retriever.invoke(q), label="Vectorstore Retriever"
    )
    vector_df.to_csv(f"{config.RESULTS_DIR}/vectorstore_retriever_results.csv", index=False)
    print_averages(vector_df, "Vectorstore Retriever")

    plot_radar(random_df, vector_df, save_path=f"{config.RESULTS_DIR}/radar_comparison.png")
    plot_bar(random_df, vector_df, save_path=f"{config.RESULTS_DIR}/bar_comparison.png")


if __name__ == "__main__":
    main()
