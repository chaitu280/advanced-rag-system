from .retriever import HybridRerankRetriever
from .generator import Generator
from .validator import CitationValidator
from .citation import (
    build_citation_metadata,
    format_citation
)
from .monitoring import RAGMonitor

from .config import MAX_GENERATION_RETRIES


class RAGPipeline:

    def __init__(
        self,
        retriever: HybridRerankRetriever,
        generator: Generator
    ):

        self.retriever = retriever
        self.generator = generator
        self.validator = CitationValidator()

        # V5 — Production Monitoring
        self.monitor = RAGMonitor()


    def build_context(
        self,
        results
    ):

        context_parts = []

        for result in results:

            document = result["document"]

            chunk_id = document.metadata.get(
                "chunk_id",
                "Unknown"
            )

            source = document.metadata.get(
                "source",
                "Unknown"
            )

            page = document.metadata.get(
                "page",
                "Unknown"
            )

            content = document.page_content

            context_parts.append(
                f"""
Chunk ID: {chunk_id}

Source: {source}

Page: {page}

Content:
{content}
"""
            )

        return "\n".join(context_parts)


    def run(
        self,
        question: str
    ):

        # =========================================
        # V5 — Start Monitoring
        # =========================================

        start_time = self.monitor.start_request()


        # =========================================
        # STEP 1 — Retrieve + Rerank
        # =========================================

        results = self.retriever.retrieve(
            question
        )


        # =========================================
        # STEP 2 — No Results
        # =========================================

        if not results:

            result = {

                "answer": (
                    "I don't know based on "
                    "the provided documents."
                ),

                "claims": [],

                "citations": [],

                "retrieved_results": [],

                "claim_validation": [],

                "validation_status": "no_results",

                "validation_errors": []
            }

            # V5 — Record monitoring data
            self.monitor.end_request(
                start_time=start_time,
                question=question,
                result=result
            )

            return result


        # =========================================
        # STEP 3 — Build Context
        # =========================================

        context = self.build_context(
            results
        )


        # =========================================
        # STEP 4 — Build Citation Metadata
        # =========================================

        citation_metadata = (
            build_citation_metadata(
                results
            )
        )


        # =========================================
        # STEP 5 — Generate + Validate
        # =========================================

        last_errors = []

        claim_validation_results = []


        for attempt in range(
            MAX_GENERATION_RETRIES + 1
        ):

            print(
                f"\nGeneration attempt "
                f"{attempt + 1}"
            )


            # =====================================
            # Generate Structured Response
            # =====================================

            response = self.generator.generate(
                question=question,
                context=context
            )


            # =====================================
            # Validate Citations
            # =====================================

            (
                is_valid,
                errors,
                claim_validation_results
            ) = self.validator.validate(

                response=response,

                retrieved_results=results
            )


            # =====================================
            # Valid Response
            # =====================================

            if is_valid:

                formatted_citations = []


                for claim in response.get(
                    "claims",
                    []
                ):

                    for citation_id in claim.get(
                        "citations",
                        []
                    ):

                        formatted = (
                            format_citation(
                                citation_id,
                                citation_metadata
                            )
                        )


                        if formatted:

                            formatted_citations.append(
                                formatted
                            )


                # Remove duplicate citations

                formatted_citations = list(
                    dict.fromkeys(
                        formatted_citations
                    )
                )


                result = {

                    "answer": response.get(
                        "answer",
                        ""
                    ),

                    "claims": response.get(
                        "claims",
                        []
                    ),

                    "citations":
                        formatted_citations,

                    # V5 — Retrieved Evidence
                    "retrieved_results":
                        results,

                    # V5 — Claim Validation
                    "claim_validation":
                        claim_validation_results,

                    "validation_status":
                        "valid",

                    "validation_errors":
                        []
                }


                # =================================
                # V5 — Record Monitoring Data
                # =================================

                self.monitor.end_request(
                    start_time=start_time,
                    question=question,
                    result=result
                )


                return result


            # =====================================
            # Invalid Response
            # =====================================

            last_errors = errors


            print(
                "\nCitation validation failed:"
            )


            for error in errors:

                print(
                    f"- {error}"
                )


        # =========================================
        # STEP 6 — All Retries Failed
        # =========================================

        result = {

            "answer": (
                "I could not generate a "
                "properly supported answer from "
                "the provided documents."
            ),

            "claims": [],

            "citations": [],

            # V5 — Retrieved Evidence
            "retrieved_results":
                results,

            # V5 — Claim Validation
            "claim_validation":
                claim_validation_results,

            "validation_status":
                "invalid",

            "validation_errors":
                last_errors
        }


        # =========================================
        # V5 — Record Monitoring Data
        # =========================================

        self.monitor.end_request(
            start_time=start_time,
            question=question,
            result=result
        )


        return result