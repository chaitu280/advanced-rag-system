import sys
from pathlib import Path
import json

from dotenv import load_dotenv


# =========================================================
# PROJECT ROOT
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(BASE_DIR))


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv(BASE_DIR / ".env")


# =========================================================
# IMPORTS
# =========================================================

from src.rag.evaluation import RAGEvaluator
from main import create_pipeline

def load_dataset():

    with open(
        "evaluation/questions.json",
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def calculate_average(results, metric):

    values = [
        result[metric]
        for result in results
    ]

    if not values:
        return 0.0

    return sum(values) / len(values)


def main():

    print("=" * 60)
    print("V5 RAG EVALUATION")
    print("=" * 60)

    dataset = load_dataset()

    pipeline = create_pipeline()

    evaluator = RAGEvaluator(
        pipeline
    )

    results = evaluator.evaluate(
        dataset
    )

    print("\n")
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    metrics = [
        "recall@5",
        "precision@5",
        "mrr",
        "citation_accuracy",
        "citation_coverage",
        "faithfulness"
    ]

    for metric in metrics:

        score = calculate_average(
            results,
            metric
        )

        print(
            f"{metric:<25}: "
            f"{score:.2%}"
        )

    print("=" * 60)

    # Save detailed results
    with open(
        "evaluation/results.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(
        "\nDetailed results saved to "
        "evaluation/results.json"
    )


if __name__ == "__main__":
    main()