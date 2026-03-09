# AI Technical Norms Assistant

A Retrieval-Augmented Generation (RAG) system designed to help users
explore, understand, and query Brazilian technical standards (such as
ABNT NBR documents) through natural language.

This project demonstrates a complete **AI-powered document intelligence
pipeline**, combining modern backend engineering, vector search, LLM
integration, and a clean user interface to build a practical knowledge
assistant.

------------------------------------------------------------------------

# Project Overview

The **AI Technical Norms Assistant** allows users to:

-   Upload technical documents (PDF)
-   Automatically extract and structure document content
-   Generate embeddings and store them in a vector database
-   Retrieve the most relevant document sections
-   Ask questions about the documents using natural language
-   Receive contextual answers grounded in the source material

The system uses **Retrieval-Augmented Generation (RAG)** to ensure
responses are based strictly on the uploaded documents.

This project focuses on **robust backend architecture**, **AI
integration**, and **traceable responses**, making it suitable for
professional or enterprise knowledge systems.

------------------------------------------------------------------------

# Key Features

### Document Ingestion Pipeline

-   Upload PDF technical documents
-   Extract text page by page using **PyMuPDF**
-   Chunk documents into structured sections
-   Generate embeddings for semantic search
-   Store metadata and vectors in **PostgreSQL + pgvector**

### AI Question Answering

-   Ask natural language questions about documents
-   Semantic search retrieves the most relevant chunks
-   LLM generates answers grounded in the retrieved context
-   Responses include traceable sources (document, page, excerpt)

### Traceable Sources

Each answer includes:

-   document title
-   document type
-   version
-   page number
-   excerpt of the source text

This allows users to **verify exactly where the information came from**.

### Logging and Debugging

The backend includes detailed logs for:

-   question processing
-   embedding generation
-   vector retrieval
-   prompt construction
-   LLM calls

This improves observability and debugging.

### Clean API Architecture

The system exposes a REST API built with **FastAPI** for:

-   document upload
-   document management
-   chat-based document querying

------------------------------------------------------------------------

# System Architecture

The architecture follows a clean layered design:

User Interface\
↓\
FastAPI REST API\
↓\
Service Layer\
↓\
Retrieval & Embedding Services\
↓\
PostgreSQL + pgvector (Vector Store)\
↓\
LLM Provider (Google Gemini)

Main components:

-   **FastAPI** -- API framework
-   **PyMuPDF** -- PDF text extraction
-   **PostgreSQL** -- relational storage
-   **pgvector** -- vector similarity search
-   **Google Gemini Flash** -- language model
-   **SQLAlchemy** -- ORM and database management

------------------------------------------------------------------------

# AI Pipeline

1.  **Document Upload**
    -   User uploads a PDF
    -   File is stored locally
2.  **Text Extraction**
    -   PDF is parsed page by page
    -   Text is structured for processing
3.  **Chunking**
    -   Content is split into smaller semantic chunks
4.  **Embedding Generation**
    -   Each chunk is converted into a vector representation
5.  **Vector Storage**
    -   Embeddings stored in PostgreSQL using pgvector
6.  **User Query**
    -   Question is converted into an embedding
7.  **Vector Search**
    -   Most relevant chunks are retrieved
8.  **LLM Generation**
    -   Gemini Flash generates an answer using retrieved context
9.  **Response**
    -   Answer + sources returned to the user

------------------------------------------------------------------------

# Example Query

User question:

What are the main requirements described in the document?

The system will:

1.  Retrieve the most relevant sections
2.  Generate a contextual answer
3.  Provide traceable sources

Example response:

The documents describe Brazilian technical standards related to academic
document formatting and document indexing.

ABNT NBR 14724:2024 defines the structure of academic works, including
elements such as cover page, introduction, development, conclusion,
references and appendices.

ABNT NBR 6034:2004 focuses specifically on how indexes should be
structured and presented within documents.

Sources include: - document - page number - excerpt

------------------------------------------------------------------------

# Tech Stack

Backend - Python - FastAPI - SQLAlchemy - PyMuPDF

AI - Google Gemini Flash - Embeddings - Retrieval-Augmented Generation

Database - PostgreSQL - pgvector

Testing - Pytest

------------------------------------------------------------------------

# Running the Project

### 1. Clone the repository

git clone `<repository-url>`{=html}\
cd rag-architecture-ai

### 2. Install dependencies

pip install -r requirements.txt

### 3. Configure environment variables

Create a `.env` file with:

DATABASE_URL=postgresql://user:password@localhost:5432/rag_architecture_ai\
GOOGLE_API_KEY=your_api_key

### 4. Start the API

uvicorn app.main:app --reload

### 5. Open the API documentation

http://127.0.0.1:8000/docs

------------------------------------------------------------------------

# Testing

Run automated tests with:

pytest -q

The project includes tests covering:

-   PDF extraction
-   ingestion pipeline
-   API endpoints
-   chat response structure

------------------------------------------------------------------------

# Future Improvements

Possible extensions for this system include:

-   reranking models for improved retrieval
-   hybrid search (keyword + vector)
-   multi-document comparison
-   streaming LLM responses
-   document version management
-   improved UI/UX
-   enterprise authentication

------------------------------------------------------------------------

# Why This Project Matters

This project demonstrates the practical application of:

-   Retrieval-Augmented Generation
-   Vector databases
-   AI-assisted knowledge systems
-   production-style backend architecture

It showcases how modern AI tools can be integrated into real systems to
improve **knowledge discovery and document understanding**.

------------------------------------------------------------------------

# Author

José Geraldo do Espírito Santo Júnior\
Brazil

LinkedIn\
https://www.linkedin.com/in/josejunior13/
