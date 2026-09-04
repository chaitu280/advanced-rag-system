from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)


def load_documents(documents_dir: Path):

    documents = []

    for file_path in documents_dir.iterdir():

        if file_path.suffix.lower() == ".pdf":

            print(f"Loading PDF: {file_path.name}")

            loader = PyPDFLoader(
                str(file_path)
            )

            documents.extend(
                loader.load()
            )

        elif file_path.suffix.lower() == ".txt":

            print(f"Loading TXT: {file_path.name}")

            loader = TextLoader(
                str(file_path),
                encoding="utf-8"
            )

            documents.extend(
                loader.load()
            )

    if not documents:
        raise ValueError(
            f"No supported documents found in {documents_dir}"
        )

    print(
        f"Loaded {len(documents)} document pages."
    )

    return documents