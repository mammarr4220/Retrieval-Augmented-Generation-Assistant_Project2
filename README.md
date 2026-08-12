This repository stores the source files for my World Health Organization Healthcare Assistant Project, which answers questions about provided HIV service-delivery guidelines using only information retrieved from the documents.

# Overview
The system combines document retrieval with an LLM to provide grounded answers with source and page citations. Users interact with the assistant through a React frontend, while Django handles the API and RAG piepline.

# Project Goal:
The goal of this project is to demonstrate how RAG can provide reliable, document-grounded answers by combining semantic search with language-model generation while giving source citations.

# Tech Stack Used:
- Frontend: React
- Backend: Django + Django REST Framework
- Large Language Model: Qwen3-4B
- Vector Database: ChromaDB
- RAG Framework: LangChain
- Document Chunking: RecursiveCharacterTextSplitter

# Features:
- Retrieves relevant information from HIV guidelines.
- Generates answers using only retrieved context.
- Provides document and page citations.
- React-based user interface.
- Django REST API for communication between frontend and RAG system.
- Persistent ChromaDB vector database.

# Architecture:
Document Ingestion -> Embedding Generation -> ChromaDB Vector Database -> User Question and Retrieval -> Context Consturction -> Answer Generation -> Django Backend -> ReactFrontend