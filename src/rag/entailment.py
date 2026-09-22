from sentence_transformers import CrossEncoder


class CitationEntailmentChecker:

    def __init__(
        self,
        model_name="cross-encoder/nli-deberta-v3-base"
    ):

        print(
            "\nLoading citation entailment model..."
        )

        self.model = CrossEncoder(
            model_name
        )

        print(
            "Citation entailment model loaded."
        )


    def check(
        self,
        claim: str,
        evidence: str
    ):

        # Premise = evidence
        # Hypothesis = claim

        scores = self.model.predict(
            [
                (
                    evidence,
                    claim
                )
            ]
        )

        scores = scores[0]

        labels = self.model.model.config.id2label

        label_scores = {
            labels[index].lower(): float(score)
            for index, score in enumerate(scores)
        }

        return label_scores


    def is_supported(
        self,
        claim: str,
        evidence: str
    ):

        scores = self.check(
            claim=claim,
            evidence=evidence
        )

        entailment_score = scores.get(
            "entailment",
            0.0
        )

        contradiction_score = scores.get(
            "contradiction",
            0.0
        )

        # Entailment must be stronger than
        # contradiction.

        supported = (
            entailment_score > contradiction_score
        )

        return supported, scores