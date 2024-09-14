# RAG Application using FAISS and HuggingFace Embeddings

This repository implements a Retrieval-Augmented Generation (RAG) system that combines information retrieval from document embeddings with generative models for answering user queries. The vector database is created using the FAISS library, and document embeddings are generated using HuggingFace models.

## Features

- **Document Embedding**: Converts documents (PDFs) into vector embeddings using a pre-trained HuggingFace model.
- **Vector Store Creation**: Stores embeddings in a FAISS vector database for efficient similarity search.
- **RAG Pipeline**: Combines document retrieval from FAISS with generation models to augment answers based on relevant information.

## RAG Pipeline
- The RAG pipeline is implemented and tested in using `rag_pipeline.ipynb`
