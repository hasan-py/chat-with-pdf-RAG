import os
import streamlit as st
import model_interaction
import re

from langchain_community.document_loaders import PDFPlumberLoader

# Directory to save uploaded PDFs
pdfs_directory = "chat-with-pdf/pdfs/"

# Ensure the directory exists
os.makedirs(pdfs_directory, exist_ok=True)


def upload_pdf(file):
    """Save the uploaded PDF to the specified directory."""
    try:
        file_path = os.path.join(pdfs_directory, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None


def load_pdf(file_path):
    """Load the content of the PDF using PDFPlumberLoader."""
    try:
        loader = PDFPlumberLoader(file_path)
        return loader.load()
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return None


# Streamlit UI
st.title("Chat with Your PDF")
uploaded_file = st.file_uploader(
    "Upload a PDF file to get started", type="pdf", accept_multiple_files=False
)


if uploaded_file:
    # Save the uploaded PDF
    file_path = upload_pdf(uploaded_file)

    if file_path:
        st.success(f"File uploaded successfully: {uploaded_file.name}")

        # Load and process the PDF
        with st.spinner("Processing PDF..."): 
            documents = load_pdf(file_path)
            #modify to send output of load_pdf() to model_interactions.py. model_interactions.py to split text and index docs using embeddings and add to vector store (split_text() and index_docs())
            if documents:
                chunked_docs = model_interaction.split_text(documents)
                model_interaction.index_docs(chunked_docs)
                st.success("PDF indexed successfully! Ask your questions below.")

        # Chat input
        question = st.chat_input("Ask a question about the uploaded PDF:")

        if question: 
            st.chat_message("user").write(question)
            with st.spinner("Retrieving relevant information..."):
                related_documents = model_interaction.retrieve_docs(question) #modify to retrieve final answer from model_interactions.py. Pass 'question'. 

                if related_documents:
                    answer = model_interaction.answer_question(question, related_documents)
                    
                    #Format answer
                    ht_matches = re.findall(r'<think>(.*?)</think>', answer, re.DOTALL)
                    ht = ht_matches[0].strip() if ht_matches else ""

                    an_parts = re.split(r'</?think>', answer)
                    an = ''.join(part.strip() for part in an_parts if part.strip() and part.strip() not in ht)

                    st.chat_message("assistant").caption(ht)
                    st.chat_message("assistant").write(an)
                else:
                    st.chat_message("assistant").write("No relevant information found.")