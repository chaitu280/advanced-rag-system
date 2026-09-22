from .metrics import (
    recall_at_k,
    precision_at_k,
    reciprocal_rank,
    citation_accuracy,
    citation_coverage,
    faithfulness,
    average_metric,
)


class RAGEvaluator:

    def __init__(
        self,
        pipeline,
        top_k=5
    ):

        self.pipeline = pipeline
        self.top_k = top_k


    def evaluate_question(
        self,
        item
    ):

        # =========================================
        # Evaluation Question
        # =========================================

        question = item["question"]

        expected_chunks = item[
            "expected_chunks"
        ]


        # =========================================
        # Run Complete RAG Pipeline
        # =========================================

        print(
            f"\nEvaluating: {question}"
        )

        result = self.pipeline.run(
            question
        )


        # =========================================
        # Retrieved Results
        # =========================================

        retrieved_results = result.get(
            "retrieved_results",
            []
        )


        retrieved_ids = []


        for item_result in retrieved_results:

            document = item_result[
                "document"
            ]

            chunk_id = document.metadata.get(
                "chunk_id"
            )


            if chunk_id:

                retrieved_ids.append(
                    chunk_id
                )


        # =========================================
        # Retrieval Metrics
        # =========================================

        recall = recall_at_k(
            retrieved_ids,
            expected_chunks,
            k=self.top_k
        )


        precision = precision_at_k(
            retrieved_ids,
            expected_chunks,
            k=self.top_k
        )


        mrr = reciprocal_rank(
            retrieved_ids,
            expected_chunks
        )


        # =========================================
        # Claims
        # =========================================

        claims = result.get(
            "claims",
            []
        )


        # =========================================
        # Citation Metrics
        # =========================================

        citation_acc = citation_accuracy(
            claims,
            retrieved_ids
        )


        citation_cov = citation_coverage(
            claims
        )


        # =========================================
        # V4 Entailment Results
        # =========================================

        validation_results = result.get(
            "claim_validation",
            []
        )


        faithfulness_score = faithfulness(
            validation_results
        )


        # =========================================
        # Return Evaluation Result
        # =========================================

        return {

            "question": question,

            "answer": result.get(
                "answer",
                ""
            ),

            "retrieved_chunks": (
                retrieved_ids
            ),

            "expected_chunks": (
                expected_chunks
            ),

            "recall@5": recall,

            "precision@5": precision,

            "mrr": mrr,

            "citation_accuracy": (
                citation_acc
            ),

            "citation_coverage": (
                citation_cov
            ),

            "faithfulness": (
                faithfulness_score
            ),

            "validation_status": (
                result.get(
                    "validation_status",
                    "unknown"
                )
            ),

            "validation_errors": (
                result.get(
                    "validation_errors",
                    []
                )
            )
        }


    def evaluate(
        self,
        dataset
    ):

        results = []


        # =========================================
        # Evaluate Every Question
        # =========================================

        for item in dataset:

            result = self.evaluate_question(
                item
            )

            results.append(
                result
            )


        return results


    def summarize(
        self,
        results
    ):

        if not results:

            return {}


        # =========================================
        # Collect Metric Values
        # =========================================

        recall_scores = [
            result["recall@5"]
            for result in results
        ]


        precision_scores = [
            result["precision@5"]
            for result in results
        ]


        mrr_scores = [
            result["mrr"]
            for result in results
        ]


        citation_accuracy_scores = [
            result["citation_accuracy"]
            for result in results
        ]


        citation_coverage_scores = [
            result["citation_coverage"]
            for result in results
        ]


        faithfulness_scores = [
            result["faithfulness"]
            for result in results
        ]


        # =========================================
        # Calculate Averages
        # =========================================

        return {

            "recall@5": average_metric(
                recall_scores
            ),

            "precision@5": average_metric(
                precision_scores
            ),

            "mrr": average_metric(
                mrr_scores
            ),

            "citation_accuracy": (
                average_metric(
                    citation_accuracy_scores
                )
            ),

            "citation_coverage": (
                average_metric(
                    citation_coverage_scores
                )
            ),

            "faithfulness": (
                average_metric(
                    faithfulness_scores
                )
            ),

            "total_questions": len(
                results
            )
        }