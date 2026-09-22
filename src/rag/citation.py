from typing import List, Dict


def build_citation_metadata(results):

    citations = {}

    for result in results:

        document = result["document"]

        chunk_id = document.metadata.get(
            "chunk_id"
        )

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        page = document.metadata.get(
            "page",
            "Unknown"
        )

        citations[chunk_id] = {
            "chunk_id": chunk_id,
            "source": source,
            "page": page
        }

    return citations


def format_citation(
    chunk_id,
    citation_metadata
):

    metadata = citation_metadata.get(
        chunk_id
    )

    if not metadata:

        return None

    source = metadata["source"]

    page = metadata["page"]

    return (
        f"[{source}, page {page}, "
        f"chunk {chunk_id}]"
    )