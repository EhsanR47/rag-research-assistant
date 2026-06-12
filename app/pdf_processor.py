import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


#VECTORSTORE_PATH = "vectorstore"


def detect_section(text):
    text_upper = text.upper()

    if "ABSTRACT" in text_upper:
        return "abstract"

    if "INTRODUCTION" in text_upper:
        return "introduction"

    if "METHOD" in text_upper:
        return "methods"

    if "RESULT" in text_upper:
        return "results"

    if "CONCLUSION" in text_upper:
        return "conclusion"

    if "REFERENCE" in text_upper:
        return "references"

    return "unknown"


def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    current_section = "unknown"

    for i, chunk in enumerate(chunks):
        section = detect_section(chunk.page_content)

        if section != "unknown":
            current_section = section

        chunk.metadata["chunk_id"] = i
        chunk.metadata["section"] = current_section

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )

    pdf_name = os.path.splitext(
        os.path.basename(pdf_path)
    )[0]

    vectorstore_path = os.path.join(
        "vectorstore",
        pdf_name
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=vectorstore_path
    )

    return len(chunks), vectorstore_path
    
    
'''
def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    current_section = "unknown"

    for i, chunk in enumerate(chunks):
        section = detect_section(
            chunk.page_content
        )

        if section != "unknown":
            current_section = section

        chunk.metadata["chunk_id"] = i
        chunk.metadata["section"] = current_section

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )

    if os.path.exists(VECTORSTORE_PATH):
        import shutil
        shutil.rmtree(VECTORSTORE_PATH)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_PATH
    )

    return len(chunks)
'''