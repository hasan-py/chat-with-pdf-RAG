import os
import pytest
import tempfile
from unittest.mock import MagicMock, patch
from pdf_rag import upload_pdf, load_pdf
from model_interaction import split_text, index_docs, retrieve_docs, answer_question
from langchain_core.documents import Document


@pytest.fixture
def sample_pdf():
    """Creates a temporary sample PDF file for testing."""
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_pdf.write(b"%PDF-1.4 Sample PDF Content")
    temp_pdf.close()
    yield temp_pdf.name
    os.remove(temp_pdf.name)


def test_upload_pdf(sample_pdf):
    """Test the upload_pdf function."""
    mock_file = MagicMock()
    mock_file.name = "test.pdf"
    mock_file.getbuffer.return_value = open(sample_pdf, "rb").read()

    saved_path = upload_pdf(mock_file)
    assert os.path.exists(saved_path)
    os.remove(saved_path)


@patch("pdf_rag.PDFPlumberLoader.load")
def test_load_pdf(mock_pdf_loader, sample_pdf):
    """Test loading a PDF."""
    mock_pdf_loader.return_value = [Document(page_content="Sample text")]
    result = load_pdf(sample_pdf)
    assert isinstance(result, list)
    assert result[0].page_content == "Sample text"


def test_split_text():
    """Test text splitting."""
    documents = [Document(page_content="Sample text that needs splitting.")]
    chunks = split_text(documents)
    assert isinstance(chunks, list)
    assert len(chunks) > 0


@patch("model_interaction.InMemoryVectorStore.add_documents")
def test_index_docs(mock_add_docs):
    """Test document indexing."""
    documents = [Document(page_content="Indexed text")]
    index_docs(documents)
    mock_add_docs.assert_called_once()


@patch("model_interaction.InMemoryVectorStore.similarity_search")
def test_retrieve_docs(mock_search):
    """Test document retrieval."""
    mock_search.return_value = [Document(page_content="Relevant text")]
    result = retrieve_docs("test query")
    assert isinstance(result, list)
    assert len(result) > 0


@patch("model_interaction.OllamaLLM.invoke")
def test_answer_question(mock_invoke):
    """Test question answering."""
    mock_invoke.return_value = "This is an answer."
    docs = [Document(page_content="Context text")]
    result = answer_question("What is this?", docs)
    assert result == "This is an answer."