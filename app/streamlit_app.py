from pdf_processor import process_pdf
import os
import streamlit as st

from rag_qa import retrieve_context, generate_answer


st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="📄",
    layout="wide"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    os.makedirs("uploaded_files", exist_ok=True)

    pdf_path = os.path.join(
        "uploaded_files",
        uploaded_file.name
    )

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.sidebar.button("Process PDF"):
        

        with st.spinner("Processing PDF..."):

            #chunk_count = process_pdf(pdf_path)
            chunk_count, vectorstore_path = process_pdf(pdf_path)
            st.session_state.vectorstore_path = vectorstore_path
        
        st.session_state.messages = []
        
        st.sidebar.success(
            f"PDF processed successfully. Created {chunk_count} chunks."
        )

st.title("📄 RAG Research Assistant")
st.write("Ask questions about your research paper using local RAG + Ollama.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("Ask a question about your PDF...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            context, sources = retrieve_context(
                question,
                st.session_state.get("vectorstore_path", "vectorstore")
            )
            answer = generate_answer(question, context)

        st.write(answer)
        if sources:
            st.markdown("### Sources")

            for source in sources:
                st.write(
                    f"Page: {source['page']} | "
                    f"Chunk: {source['chunk_id']} | "
                    f"Section: {source['section']}"
                )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )