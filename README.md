# Chat with Your PDFs using RAG

This project allows you to upload a PDF and ask questions about its content using **Deepseek R1** via **Ollama**. The application processes PDFs, extracts text, indexes them into a vector store, and retrieves relevant context to generate concise answers.

## Features

- 📂 **Upload a PDF**: Select a PDF file to process.
- 🔍 **Text Extraction & Indexing**: Extracts content and indexes it for efficient search.
- 💡 **Question-Answering**: Ask questions related to the PDF content and get relevant answers.
- 🚀 **Powered by Ollama & LangChain**: Uses `Deepseek R1` for embeddings and responses.

## Installation

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com) installed
- Dependencies installed via pip

### Setup

1. Clone this repository:

   ```sh
   git clone https://github.com/hasan-py/chat-with-pdf-RAG.git
   cd chat-with-pdf-RAG
   ```

   Activate your python env and install the dependencies.

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```sh
   streamlit run pdf_rag.py
   ```

## How It Works

1. **Upload a PDF**: Use the UI to upload a document.
2. **Processing**: The app extracts text and chunks it for indexing.
3. **Ask Questions**: Enter a question in the chat box.
4. **Get Answers**: The system retrieves relevant text and responds concisely.

## How to change model?

To change the model used for inference, you can modify the `LLM` variable in the `pdf_rag.py` file. The `LLM` variable is initialized with the `deepseek-r1:8b` model by default. You can replace it with any other model supported by `Ollama`.

## File Structure

```
chat-with-pdf/
│── pdfs/                   # Directory for uploaded PDFs
│── pdf_rag.py              # Main Streamlit app
│── requirements.txt        # Dependencies
│── README.md               # Documentation
│── test_pdf_rag.py         # Unit Test 
```

## Technologies Used

- **Python**
- **Streamlit** (for UI)
- **LangChain** (for text processing)
- **Ollama** (for LLM inference)
- **PDFPlumber** (for PDF extraction)

## Contributing

Feel free to submit issues and PRs to improve the project! And follow this steps:

- Before submitting PRs, please update the corresponding test cases. 
- Please attach a screen recording video to the PR description showing that all functionality is working properly.


## Acknowledgments

Special thanks to the creators of **LangChain**, **Ollama**, **Streamlit** and the **community** for enabling this functionality.
