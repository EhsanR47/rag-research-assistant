import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


PDF_PATH = "data/papers/sample.pdf"
VECTORSTORE_PATH = "vectorstore"


def load_pdf(pdf_path):
    # Load PDF pages as LangChain documents
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents


def detect_section(text):
    text_upper = text.upper()

    if "ABSTRACT" in text_upper:
        return "abstract"
    if "1. INTRODUCTION" in text_upper or "INTRODUCTION" in text_upper:
        return "introduction"
    if "2. DATA" in text_upper or "DATA" in text_upper:
        return "data"
    if "3. METHODS" in text_upper or "METHODS" in text_upper:
        return "methods"
    if "4. C III" in text_upper or "RESULTS" in text_upper:
        return "results"
    if "5. CONCLUSIONS" in text_upper or "CONCLUSIONS" in text_upper:
        return "conclusion"
    if "REFERENCES" in text_upper:
        return "references"

    return "unknown"


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)

    current_section = "unknown"

    for i, chunk in enumerate(chunks):
        detected_section = detect_section(chunk.page_content)

        if detected_section != "unknown":
            current_section = detected_section

        chunk.metadata["chunk_id"] = i
        chunk.metadata["section"] = current_section

    return chunks


def create_vectorstore(chunks):
    # Create embeddings using a local sentence-transformer model
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )

    # Store document chunks in ChromaDB
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_PATH
    )

    return vectorstore


def main():
    if not os.path.exists(PDF_PATH):
        print(f"PDF file not found: {PDF_PATH}")
        print("Please put your PDF file inside data/papers/ and rename it to sample.pdf")
        return

    print("Loading PDF...")
    documents = load_pdf(PDF_PATH)
    print(f"Loaded pages: {len(documents)}")

    print("\nFirst pages preview:\n")

    for i, doc in enumerate(documents[:3]):
        print("=" * 80)
        print(f"Page {i}")
        print("-" * 80)
        print(doc.page_content[:1000])
        print()

    print("Splitting documents...")
    chunks = split_documents(documents)
    print(f"Created chunks: {len(chunks)}")
    print("\nChunk section preview:\n")

    for chunk in chunks[:20]:
        print(
            f"Chunk ID: {chunk.metadata.get('chunk_id')} | "
            f"Page: {chunk.metadata.get('page')} | "
            f"Section: {chunk.metadata.get('section')}"
        )

    print("Creating vectorstore...")
    create_vectorstore(chunks)

    print("Done! Vector database created successfully.")


if __name__ == "__main__":
    main()