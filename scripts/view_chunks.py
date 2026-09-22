import sys
from pathlib import Path


# =================================================
# Add project root to Python path
# =================================================

BASE_DIR = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(BASE_DIR)
)


# =================================================
# Imports
# =================================================

from src.rag.config import VECTORSTORE_DIR
from src.rag.bm25 import BM25Retriever


def main():

    # =============================================
    # BM25 Index
    # =============================================

    bm25_path = (
        VECTORSTORE_DIR / "bm25.pkl"
    )


    print("=" * 70)

    print(
        "AVAILABLE DOCUMENT CHUNKS"
    )

    print("=" * 70)


    # =============================================
    # Load BM25
    # =============================================

    print(
        f"\nLoading BM25 index:"
        f"\n{bm25_path}"
    )


    retriever = BM25Retriever.load(
        bm25_path
    )


    documents = retriever.documents


    print(
        f"\nTotal chunks: "
        f"{len(documents)}"
    )


    # =============================================
    # Display Chunks
    # =============================================

    for index, document in enumerate(
        documents
    ):

        chunk_id = document.metadata.get(
            "chunk_id",
            f"chunk_{index}"
        )


        source = document.metadata.get(
            "source",
            "Unknown"
        )


        page = document.metadata.get(
            "page",
            "Unknown"
        )


        print(
            "\n" + "=" * 70
        )


        print(
            f"Chunk ID : {chunk_id}"
        )


        print(
            f"Source   : {source}"
        )


        print(
            f"Page     : {page}"
        )


        print(
            "=" * 70
        )


        print(
            document.page_content
        )


        print()


# =================================================
# Entry Point
# =================================================

if __name__ == "__main__":

    main()