# RAG QA Bot

This application allows users to upload PDF files, create a vector database from the document using open-source HuggingFace embeddings, and ask questions related to the PDF content using a Retrieval-Augmented Generation approach. The app integrates with LangChain Framework, OpenAI's LLM and HuggingFace embeddings.

## Features
- Upload a PDF file and save it locally. Later we can create the API to delete the old files.
- Create a vector database from the PDF's content using HuggingFace model `sentence-transformers/all-mpnet-base-v2`
- Ask questions about the PDF content.
- View the context used for answering the questions that is toggleable via a checkbox.
- The POC of RAG pipeline is tested in using `rag_pipeline.ipynb`

## Screenshots of the RAG app
- The main screen when you open app
![first](https://github.com/user-attachments/assets/4c720ae0-14ba-4829-9dff-e0ec37621aa2)

-  you upload the pdf
![second](https://github.com/user-attachments/assets/27e85cc6-4b0f-4969-ba6d-35217f6699aa)

- After you create embeddings
![third](https://github.com/user-attachments/assets/911ebcc3-f220-429e-bd78-b5cad3555942)

- QA Screen
![Screenshot from 2024-09-14 20-05-29](https://github.com/user-attachments/assets/ade68bec-8704-43ae-81cb-f29d47806ec0)
  
 

## Installation
### Clone the Repository
```bash
git clone https://github.com/yourusername/rag-qa-bot.git
cd rag-qa-bot
```
### Set up a Virtual Environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### The main packages are:
- `streamlit`: For the web UI.
- `PyPDFLoader`: To extract content from PDF files.
- `langchain`: For embeddings, document chunking, and question-answering.
- `faiss-cpu`: For vector store creation and retrieval.
- `openai`: To integrate with OpenAI's language models.

## Configuration

We provide a `config.json` file in the root directory, this will allow you to select the models at your choice, with the following details:

```json
{
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "openai_api_key": "your_openai_api_key",
  "openai_model": "gpt-3.5-turbo",
  "vector_db_path": "./vector_store"
}
```
## Usage

### Running the App
To launch the Streamlit web app, run the following command in your terminal:

```bash
streamlit run main.py
```
browse the url `http://localhost:8501/`

## How It Works

1. **PDF Upload**: The user uploads a PDF file using  the Streamlit file uploader.
2. **Document Chunking**: The PDF content is split into manageable chunks using the `RecursiveCharacterTextSplitter` api fo LangChain.
3. **Embeddings Generation**: The chunks are passed through a HuggingFace embedding model to generate embeddings.
4. **Vector Store Creation**: The embeddings are stored in a FAISS-based vector store, which is then saved locally.
5. **Question Answering**: When a user asks a question, the system retrieves the relevant context from the vector store and generates an answer using OpenAI's LLM.

## Project Structure

```bash
|-- src/
|   |-- utils.py             # Helper functions such as file-saving logic
    |-- rag_application.py   # Class to implement the RAG pipeline
|-- main.py                  # Main Streamlit app file
|-- requirements.txt         # List of required dependencies
|-- config.json              # Configuration file
|-- rag_pipeline.ipynb       # Test and POC the RAG pipeline
```

