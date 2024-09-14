import streamlit as st
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from src.utils import save_uploaded_file

class RAGQABot:
    def __init__(self, configuration_path):
        print('CONTRUCTOR')
        st.title("RAG App: Upload PDF, Create Vector Store, and Perform QA")

        self.file_path = None
        
        with open(configuration_path) as f:
            self.config = json.load(f)

        self.embedding_client = HuggingFaceEmbeddings(model_name=self.config['embedding_model'])
        # print(config["embedding_model"])

        # Initialize session state variables
        if "file_uploaded" not in st.session_state:
            st.session_state.file_uploaded = False
        if "vector_db_created" not in st.session_state:
            st.session_state.vector_db_created = False

    def upload_file(self):
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
        if uploaded_file:
            self.file_path = save_uploaded_file(uploaded_file)
            st.session_state.file_uploaded = True
            st.success(f"File uploaded and saved at {self.file_path} successfully!!")

    def create_vector_db(self):
        # Disable 'Create Vector DB' button until the file is uploaded
        if st.session_state.file_uploaded:
            if st.button('Create Vector DB'):
                if self.file_path:
                    print("File is uploaded", self.file_path)
                    loader = PyPDFLoader(self.file_path)
                    document = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=100
                    )
                    chunks = text_splitter.split_documents(document)

                    # Create and save vector store
                    vectorstore = FAISS.from_documents(chunks, self.embedding_client)
                    vectorstore.save_local(self.config['vector_db_path'])
                    st.session_state.vector_db_created = True
                    st.success("Vector store created and saved successfully!")
                else:
                    st.error("Upload the file first.")
        else:
            st.error("Please upload a file to enable vector DB creation.")

    def qa_on_vector_store(self):
        # Disable text input tilll vectordb is created 
        if st.session_state.vector_db_created:
            query = st.text_input("Ask a question about the PDF:")
            if query:
                # Initialize llm model, adn we are using openai
                llm = ChatOpenAI(
                    api_key=self.config['openai_api_key'],
                    model=self.config['openai_model']
                )

                # Load already created embeddings
                vectorstore = FAISS.load_local(self.config['vector_db_path'], self.embedding_client, allow_dangerous_deserialization=True)

                # do some prompt engineering
                system_prompt = (
                    """
                    You are an assistant for question-answering tasks.
                    Use the following pieces of retrieved context to answer
                    the question. Please stick with the context and embeddings given to you. If you don't know the answer, say that you
                    don't know. Keep the answer concise.
                    \n\n
                    {context}
                    """
                )
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system_prompt),
                        ("human", "{input}"),
                    ]
                )

                # Perform retrieval
                retriever = vectorstore.as_retriever()

                # Chain the pieces together
                question_answer_chain = create_stuff_documents_chain(llm, prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                response = rag_chain.invoke({"input": query})
                # print(response)

                st.write(f"Answer: {response['answer']}")
                show_context = st.checkbox("Show Context")
                if show_context:
                    st.write(f"Context: {response['context'][0]}")
        else:
            st.error("Please create the vector database first.")
