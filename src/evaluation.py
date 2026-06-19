"""
RAGAS-based evaluation of a retrieval strategy.

NOTE: the judge model defaults to the same model used for generation
(see config.JUDGE_MODEL). This introduces self-evaluation bias -- for a
fairer comparison, point JUDGE_MODEL at an independent model (e.g. an
OpenAI model) once you have API access.
"""
import pandas as pd
from datasets import Dataset
from langchain_community.llms import Ollama
from ragas import evaluate, RunConfig
from ragas.metrics import answer_relevancy, faithfulness, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper

from src import config


def run_condition(questions, ground_truths, retrieve_fn, label=""):
    """
    retrieve_fn: callable(question) -> list[Document]
    Generates an answer per question using the given retrieval function,
    then scores everything with RAGAS.
    """
    retrieved_contexts = []
    generated_answers = []
    llm = Ollama(model=config.GENERATOR_MODEL)

    print(f"Generating responses for: {label}")
    for i, question in enumerate(questions, 1):
        print(f"Processing question {i}/{len(questions)}")
        docs = retrieve_fn(question)
        context = "\n\n".join(doc.page_content for doc in docs)
        retrieved_contexts.append([context])
        answer = llm.invoke(question + "\nContext:\n" + context)
        generated_answers.append(answer)

    ragas_dataset = Dataset.from_dict({
        "question": questions,
        "answer": generated_answers,
        "contexts": retrieved_contexts,
        "ground_truth": ground_truths,
    })

    judge = LangchainLLMWrapper(Ollama(model=config.JUDGE_MODEL))
    run_config = RunConfig(timeout=180, max_workers=8)

    print(f"Running RAGAS evaluation for: {label}")
    results = evaluate(
        dataset=ragas_dataset,
        llm=judge,
        metrics=[answer_relevancy, faithfulness, context_precision, context_recall],
        run_config=run_config,
    )
    return results.to_pandas()


def print_averages(results_df, label=""):
    print(f"\nAverage scores ({label}):")
    for metric in ["answer_relevancy", "faithfulness", "context_precision", "context_recall"]:
        if metric in results_df.columns:
            print(f"{metric}: {results_df[metric].mean():.4f}")
