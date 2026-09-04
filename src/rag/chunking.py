from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from .config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ],
        add_start_index=True
    )

    chunks = splitter.split_documents(
        documents
    )

    for index, chunk in enumerate(chunks):

        chunk.metadata["chunk_id"] = (
            f"chunk_{index}"
        )

    print(
        f"Created {len(chunks)} chunks."
    )

    return chunks