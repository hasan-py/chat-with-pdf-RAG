import os
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

LLM = "deepseek-r1:8b"

# Prompt template for answering questions
template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

# Directory to save uploaded PDFs
pdfs_directory = "chat-with-pdf/pdfs/"

# Ensure the directory exists
os.makedirs(pdfs_directory, exist_ok=True)

# Initialize embeddings and model
embeddings = OllamaEmbeddings(model=LLM)
model = OllamaLLM(model=LLM)

# Initialize vector store
vector_store = None


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


def split_text(documents):
    """Split the documents into smaller chunks for indexing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    return text_splitter.split_documents(documents)


def index_docs(documents):
    """Index the documents in the vector store."""
    global vector_store
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents)


def retrieve_docs(query):
    """Retrieve relevant documents based on the query."""
    return vector_store.similarity_search(query)


def answer_question(question, documents):
    """Generate an answer to the question using the retrieved documents."""
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})


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
            if documents:
                chunked_documents = split_text(documents)
                index_docs(chunked_documents)
                st.success("PDF indexed successfully! Ask your questions below.")

        # Chat input
        question = st.chat_input("Ask a question about the uploaded PDF:")

        if question:
            st.chat_message("user").write(question)

            with st.spinner("Retrieving relevant information..."):
                related_documents = retrieve_docs(question)
                if related_documents:
                    answer = answer_question(question, related_documents)
                    st.chat_message("assistant").write(answer)
                else:
                    st.chat_message("assistant").write("No relevant information found.")
