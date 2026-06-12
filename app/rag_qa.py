from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


VECTORSTORE_PATH = "vectorstore"


def load_vectorstore(vectorstore_path="vectorstore"):
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )

    vectorstore = Chroma(
        #persist_directory=VECTORSTORE_PATH,
        persist_directory=vectorstore_path,
        embedding_function=embeddings
    )

    return vectorstore

def classify_question(question):
    question = question.lower()

    if any(word in question for word in [
        "author",
        "authors",
        "written by"
    ]):
        return "author"

    if any(word in question for word in [
        "main goal",
        "objective",
        "purpose",
        "summary",
        "abstract"
    ]):
        return "abstract"

    if any(word in question for word in [
        "method",
        "methods",
        "dataset",
        "data",
        "observation",
        "observations"
    ]):
        return "methods"

    if any(word in question for word in [
        "result",
        "results",
        "finding",
        "findings",
        "conclusion"
    ]):
        return "results"

    if any(word in question for word in [
        "reference",
        "references",
        "citation"
    ]):
        return "references"

    return "general"

def retrieve_context(question, vectorstore_path="vectorstore"):
    vectorstore = load_vectorstore(vectorstore_path)
    question_type = classify_question(question)

    print(f"\nDetected question type: {question_type}\n")

    question_lower = question.lower()

    # If the question asks about authors, use PDF metadata first
    if "author" in question_lower:
        docs = vectorstore.similarity_search(question, k=3)

        for doc in docs:
            authors = doc.metadata.get("author")
            if authors:
                print("\nUsing PDF metadata for authors.\n")
                return f"Authors: {authors}", [
                    {
                        "page": "N/A",
                        "chunk_id": "N/A",
                        "section": "PDF Metadata"
                    }
                ]

    # For general paper-level questions, search only early pages
    if question_type == "abstract":
        docs = vectorstore.similarity_search(
            question,
            k=5,
            filter={"section": "abstract"}
        )

    elif question_type == "methods":
        docs = vectorstore.similarity_search(
            question,
            k=5,
            filter={"section": "data"}
        )

    elif question_type == "results":
        docs = vectorstore.similarity_search(
            question,
            k=5,
            filter={"section": "results"}
        )

    elif question_type == "references":
        docs = vectorstore.similarity_search(
            question,
            k=5,
            filter={"section": "references"}
        )

    else:
        docs = vectorstore.similarity_search(
            question,
            k=5
        )

    print("\nRetrieved context:\n")

    for i, doc in enumerate(docs, start=1):
        print("=" * 80)
        print(f"Result {i}")
        print(f"Page: {doc.metadata.get('page', 'Unknown')}")
        print(f"Chunk ID: {doc.metadata.get('chunk_id', 'Unknown')}")
        print("-" * 80)
        print(doc.page_content[:500])
        print()

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    sources = []

    for doc in docs:
        sources.append({
            "page": doc.metadata.get("page", "Unknown"),
            "chunk_id": doc.metadata.get("chunk_id", "Unknown"),
            "section": doc.metadata.get("section", "Unknown")
        })

    return context, sources


def generate_answer(question, context):
    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )

    prompt = f"""
You are a helpful research assistant.

Answer ONLY using the provided context.

If the answer is not in the context, say:
"I could not find the answer in the document."

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    return response.content


def main():
    print("RAG Research Assistant")
    print("Type 'exit' to stop.\n")

    while True:
        question = input("Question: ")

        if question.lower() == "exit":
            break

        context = retrieve_context(question)

        answer = generate_answer(
            question,
            context
        )

        print("\nAnswer:\n")
        print(answer)
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()