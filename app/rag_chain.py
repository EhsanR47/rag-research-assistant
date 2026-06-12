from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


VECTORSTORE_PATH = "vectorstore"


def load_vectorstore():
    # Load the same embedding model used during ingestion
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )

    # Load the existing Chroma vector database
    vectorstore = Chroma(
        persist_directory=VECTORSTORE_PATH,
        embedding_function=embeddings
    )

    return vectorstore


def search_documents(question, k=10):
    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 50,
            "lambda_mult": 0.5
        }
    )

    documents = retriever.invoke(question)

    return documents

def main():
    print("RAG Research Assistant")
    print("Ask a question about your PDF.")
    print("Type 'exit' to stop.\n")

    while True:
        question = input("Question: ")

        if question.lower() == "exit":
            break

        results = search_documents(question)

        print("\nTop relevant chunks:\n")

        for i, document in enumerate(results, start=1):
            print("=" * 80)
            print(f"Retrieved Result {i}")

            print(
                f"Page: {document.metadata.get('page', 'Unknown')}"
            )

            print(
                f"Chunk ID: {document.metadata.get('chunk_id', 'Unknown')}"
            )

            print("\nMetadata:")
            print(document.metadata)

            print("-" * 80)
            print(document.page_content[:1200])
            print()

        print("=" * 80)
        print()


if __name__ == "__main__":
    main()