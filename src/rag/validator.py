from .entailment import CitationEntailmentChecker


class CitationValidator:

    def __init__(self):

        self.entailment_checker = (
            CitationEntailmentChecker()
        )


    def validate(
        self,
        response,
        retrieved_results
    ):

        errors = []

        claim_validation = []

        claims = response.get(
            "claims",
            []
        )


        # =========================================
        # No Claims
        # =========================================

        if not claims:

            errors.append(
                "No claims were returned."
            )

            return (
                False,
                errors,
                claim_validation
            )


        # =========================================
        # Retrieved Chunk Mapping
        # =========================================

        retrieved_documents = {}

        for result in retrieved_results:

            document = result["document"]

            chunk_id = document.metadata.get(
                "chunk_id"
            )

            if chunk_id:

                retrieved_documents[
                    chunk_id
                ] = document


        retrieved_chunk_ids = set(
            retrieved_documents.keys()
        )


        # =========================================
        # Validate Each Claim
        # =========================================

        for claim in claims:

            text = claim.get(
                "claim"
            )

            citations = claim.get(
                "citations",
                []
            )


            # -------------------------------------
            # Claim Text
            # -------------------------------------

            if not text:

                errors.append(
                    "Claim has no text."
                )

                claim_validation.append(
                    False
                )

                continue


            # -------------------------------------
            # Citation Exists
            # -------------------------------------

            if not citations:

                errors.append(
                    f"Claim has no citation: {text}"
                )

                claim_validation.append(
                    False
                )

                continue


            claim_supported = False


            # =====================================
            # Check Each Citation
            # =====================================

            for citation_id in citations:


                # ---------------------------------
                # Citation ID Validation
                # ---------------------------------

                if citation_id not in retrieved_chunk_ids:

                    errors.append(
                        f"Invalid citation "
                        f"{citation_id} "
                        f"for claim: {text}"
                    )

                    continue


                # ---------------------------------
                # Get Evidence
                # ---------------------------------

                document = retrieved_documents[
                    citation_id
                ]

                evidence = document.page_content


                # ---------------------------------
                # Entailment Check
                # ---------------------------------

                supported, scores = (
                    self.entailment_checker.is_supported(
                        claim=text,
                        evidence=evidence
                    )
                )


                print(
                    "\n[ENTAILMENT]"
                )

                print(
                    f"Claim: {text}"
                )

                print(
                    f"Citation: {citation_id}"
                )

                print(
                    f"Scores: {scores}"
                )

                print(
                    f"Supported: {supported}"
                )


                if supported:

                    claim_supported = True

                    break


            # =====================================
            # Claim Result
            # =====================================

            claim_validation.append(
                claim_supported
            )


            if not claim_supported:

                errors.append(
                    f"Claim is not supported "
                    f"by its citations: {text}"
                )


        # =========================================
        # Final Validation
        # =========================================

        if errors:

            return (
                False,
                errors,
                claim_validation
            )


        return (
            True,
            [],
            claim_validation
        )