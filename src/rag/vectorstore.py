from pathlib import Path

from langchain_community.vectorstores import FAISS


def create_vectorstore(
    documents,
    embeddings,
    save_path: Path
):

    print("Creating FAISS vector store...")

    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )

    save_path.mkdir(
        parents=True,
        exist_ok=True
    )

    vectorstore.save_local(
        str(save_path)
    )

    print(
        f"Vector store saved to {save_path}"
    )

    return vectorstore


def load_vectorstore(
    embeddings,
    save_path: Path
):

    if not save_path.exists():

        raise FileNotFoundError(
            f"Vector store not found at {save_path}"
        )

    vectorstore = FAISS.load_local(
        str(save_path),
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore