from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

import re

# Initialize embeddings and model
embeddings = OllamaEmbeddings(model='nomic-embed-text')
model = OllamaLLM(model="deepseek-r1")

# Initialize vector store
vector_store = None

def load_pdf(file_path):
    """Load the content of the PDF using PDFPlumberLoader."""
    try:
        loader = PDFPlumberLoader(file_path)
        return loader.load()
    except Exception as e:
        print(f"Error loading PDF: {e}")
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


# Retrieve simlarity search results
def retrieve_docs(query):
    """Retrieve relevant documents based on the query."""
    return vector_store.similarity_search(query)

# Returns LLMs answer based on question and vector search query return 
def answer_question(question, related_documents):
    """Generate an answer to the question using the retrieved documents."""
    context = "\n\n".join([doc.page_content for doc in related_documents])

    # Prompt template for answering questions
    template = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Question: {question} 
    Context: {context} 
    Answer:
    """

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    return chain.invoke({"question": question, "context": context})


if __name__ == "__main__":

    # doc_path = "C:/Machine Learning/RAG/chat-with-pdf-RAG/chat-with-pdf/pdfs/transcripts_3.pdf"

    # documents = load_pdf(doc_path)

    # chunked_docs = split_text(documents)
    # for elem in chunked_docs:
    #     print(type(elem))

    # index_docs(chunked_docs)
    # print('document embedded')

    # question = "hi document"
    # answer = answer_question(question, retrieve_docs(question))

    # ht_matches = re.findall(r'<think>(.*?)</think>', answer, re.DOTALL)
    # ht = ht_matches[0].strip() if ht_matches else ""

    # an_parts = re.split(r'</?think>', answer)
    # an = ''.join(part.strip() for part in an_parts if part.strip() and part.strip() not in ht)
    # print(f"html: {ht}")
    # print(f"answer: {an}")



