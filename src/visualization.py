"""
Plotting helpers for comparing retrieval strategies on RAGAS metrics.
"""
import numpy as np
import matplotlib.pyplot as plt

METRICS = ["Context Precision", "Context Recall", "Answer Relevancy", "Faithfulness"]
KEYS = ["context_precision", "context_recall", "answer_relevancy", "faithfulness"]


def _scores(df):
    return [df[k].mean() for k in KEYS]


def plot_radar(random_df, vectorstore_df, save_path=None):
    random_scores = _scores(random_df)
    vectorstore_scores = _scores(vectorstore_df)

    angles = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False).tolist()
    angles += angles[:1]
    random_scores += random_scores[:1]
    vectorstore_scores += vectorstore_scores[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, random_scores, "o-", linewidth=2, label="Random Retriever", color="red")
    ax.fill(angles, random_scores, alpha=0.25, color="red")
    ax.plot(angles, vectorstore_scores, "o-", linewidth=2, label="Vectorstore Retriever", color="blue")
    ax.fill(angles, vectorstore_scores, alpha=0.25, color="blue")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    rmax = ax.get_ylim()[1]
    label_r = rmax * 1.08
    for angle, label in zip(angles[:-1], METRICS):
        rotation = 90 if label in ["Answer Relevancy", "Context Precision"] else 0
        ax.text(angle, label_r, label, fontsize=12, ha="center", va="center",
                 rotation=rotation, rotation_mode="anchor")

    plt.title("RAG Metrics Comparison", y=1.1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_bar(random_df, vectorstore_df, save_path=None):
    random_scores = _scores(random_df)
    vectorstore_scores = _scores(vectorstore_df)

    x = np.arange(len(METRICS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, random_scores, width, label="Random Retriever", color="red", alpha=0.7)
    bars2 = ax.bar(x + width / 2, vectorstore_scores, width, label="Vectorstore Retriever", color="blue", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(METRICS)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("RAG Metrics Comparison")
    ax.legend()

    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
