# Project Specification

## AI Technical Norms Assistant for Architecture and Safety

Version: 1.0\
Status: MVP Specification\
Author: Project Owner\
Purpose: Specification optimized for AI coding assistants (Copilot,
Codex, etc.)

------------------------------------------------------------------------

# 1. Project Overview

## Objective

Build an AI-powered assistant capable of answering technical questions
related to **architectural safety regulations** using a collection of
**PDF documents** such as:

-   Architecture technical manuals
-   Fire department regulations
-   Safety regulations
-   Building codes
-   Civil code sections related to construction
-   Accessibility standards
-   Safety and prevention documentation

The system will ingest these documents, index them, and allow users to
ask natural language questions.

The assistant must return **technically grounded answers** based on the
document collection.

------------------------------------------------------------------------

# 2. Core Concept

The system will implement **Retrieval Augmented Generation (RAG)**.

Pipeline:

PDF Documents\
→ Text Extraction\
→ Chunking\
→ Embeddings\
→ Vector Storage\
→ Semantic Retrieval\
→ LLM Response Generation

------------------------------------------------------------------------

# 3. Target Users

Primary users:

-   Architects
-   Civil engineers
-   Safety engineers
-   Fire safety consultants
-   Architecture students
-   Technical consultants

Main use cases:

-   Understanding regulations
-   Finding specific requirements
-   Clarifying technical rules
-   Locating references inside standards

------------------------------------------------------------------------

# 4. System Goals

The AI assistant must:

1.  Ingest PDF documents
2.  Extract and structure document text
3.  Split text into semantic chunks
4.  Generate embeddings for chunks
5.  Store chunks in a vector-enabled database
6.  Retrieve relevant chunks when a user asks a question
7.  Use an LLM to generate answers grounded in the retrieved content
8.  Return answers with references to sources

------------------------------------------------------------------------

# 5. Key Requirements

## Functional Requirements

### Document Upload

The system must allow uploading PDF documents.

Uploaded files should:

-   be stored locally
-   be registered in the database
-   be processed for indexing

### Document Processing

Processing steps:

1.  Extract text
2.  Identify page numbers
3.  Split text into chunks
4.  Generate embeddings
5.  Store chunks and metadata

### Question Answering

Users must be able to ask questions such as:

-   "What are the requirements for emergency stairs?"
-   "What is the minimum corridor width for evacuation?"
-   "What safety rules apply to multi‑story buildings?"

The system must:

-   retrieve relevant chunks
-   generate a technical explanation
-   cite sources

### Response Format

Each answer must contain:

1.  Objective answer
2.  Technical explanation
3.  Source references

Example:

Answer: The regulation requires a minimum width of X meters.

Explanation: This ensures safe evacuation in emergency conditions.

Sources: Document: Fire Safety Instruction 01\
Page: 14

------------------------------------------------------------------------

# 6. Non‑Functional Requirements

The system must:

-   run locally for development
-   use only free tools for MVP
-   support expansion later
-   maintain modular architecture
-   be easy for AI coding assistants to maintain

------------------------------------------------------------------------

# 7. Technology Stack

## Backend

Python 3.11+

Framework: FastAPI

## AI Orchestration

LangChain

## LLM

Google Gemini API (Free Tier)

## Embeddings

Gemini embeddings or open-source embeddings

## Database

PostgreSQL

Vector support:

pgvector extension

## PDF Processing

Preferred libraries:

PyMuPDF\
or pypdf

## Storage

Local file storage for PDFs

------------------------------------------------------------------------

# 8. System Architecture

High level architecture:

User Question\
→ API (FastAPI)\
→ Retrieval Service\
→ Vector Search\
→ Context Retrieval\
→ LLM Generation\
→ Answer Response

Document ingestion:

PDF Upload\
→ Text Extraction\
→ Chunk Generation\
→ Embedding Creation\
→ Database Storage

------------------------------------------------------------------------

# 9. Project Structure

Recommended project layout:

backend/ app/ api/ routes_chat.py routes_upload.py

        core/
            config.py
            database.py

        models/
            document.py
            chunk.py

        repositories/
            document_repository.py
            chunk_repository.py

        services/
            pdf_service.py
            chunk_service.py
            embedding_service.py
            retrieval_service.py
            chat_service.py

        main.py

storage/ pdfs/

sql/ init.sql

project-spec.md

requirements.txt

README.md

------------------------------------------------------------------------

# 10. Database Design

## Table: documents

Fields:

id\
title\
file_path\
uploaded_at\
document_type\
version

## Table: chunks

Fields:

id\
document_id\
page_number\
content\
embedding vector\
created_at

------------------------------------------------------------------------

# 11. Core Modules

## PDF Service

Responsibilities:

-   load PDF
-   extract text
-   return text with page numbers

## Chunk Service

Responsibilities:

-   split text into chunks
-   maintain semantic integrity

## Embedding Service

Responsibilities:

-   generate embeddings
-   call Gemini embedding model

## Retrieval Service

Responsibilities:

-   receive question
-   compute embedding
-   query vector database
-   return relevant chunks

## Chat Service

Responsibilities:

-   assemble prompt
-   inject context
-   call Gemini
-   format response

------------------------------------------------------------------------

# 12. Prompt Engineering Strategy

The assistant must follow strict rules:

1.  Only answer using retrieved context
2.  Never invent technical regulations
3.  If no context is found, respond accordingly
4.  Always cite source references
5.  Explain technical language when possible

------------------------------------------------------------------------

# 13. Example Prompt Template

System Prompt:

You are a technical assistant specialized in architecture and safety
regulations.

Rules:

-   Answer only using the provided documents.
-   Do not invent information.
-   Cite document and page when possible.
-   Explain technical concepts clearly.

User Question:

{question}

Context:

{retrieved_chunks}

------------------------------------------------------------------------

# 14. Development Roadmap

## Phase 1 --- Local MVP

Goals:

-   FastAPI backend
-   PDF upload
-   text extraction
-   chunk generation
-   embeddings
-   vector search
-   AI responses

## Phase 2 --- Improved Retrieval

Add:

-   metadata filters
-   better chunking
-   multi-document retrieval

## Phase 3 --- Production Ready

Add:

-   authentication
-   document versioning
-   UI interface
-   monitoring

------------------------------------------------------------------------

# 15. Success Criteria

The MVP is considered successful if:

-   PDFs can be uploaded
-   documents are indexed
-   users can ask questions
-   answers reference sources
-   system runs locally

------------------------------------------------------------------------

# 16. Future Improvements

Potential improvements:

-   OCR support
-   table extraction
-   multi-language support
-   document comparison
-   regulation conflict detection
-   automated updates of norms

------------------------------------------------------------------------

# End of Specification
