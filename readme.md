# 🚗 Mecanic-IA: Multimodal Automotive RAG

**Mecanic-IA** is an advanced Retrieval-Augmented Generation (RAG) system designed to assist mechanics by querying technical automotive manuals (specifically Dacia). It uses a **Parent-Child Chunking** strategy and **Multimodal capabilities** (Text + Vision) to provide precise, grounded answers from complex PDF documentation.

## 🌟 Key Features

*   **⚡ Parent-Child Retrieval**: Preserves document context by indexing small "child" chunks for search but retrieving full "parent" sections for the LLM.
*   **🖼️ Multimodal Vision**: Automatically extracts images from PDFs, captions them using a local Vision LLM (`llava-phi3`), and makes them searchable alongside text.
*   **🧠 Local & Private**: Powered entirely by **Ollama** (Llama 3.2 for text, LLaVA for vision) ensuring data privacy.
*   **🧩 Modular Architecture**: Built with a clean, standard FastAPI structure (`core`, `api`, `services`).
*   **🔍 Hybrid Search**: Combines dense vector retrieval (ChromaDB) with Cross-Encoder reranking for high accuracy.

---

## 🏗️ Architecture

### 1. Ingestion Pipeline
1.  **PDF Parsing**: Extracts text and detects headers to identify logical sections (Parent Chunks).
2.  **Child Splitting**: Splits parents into smaller, overlapping chunks (Child Chunks) for better vector search performance.
3.  **Vision Processing**: Extracts images, generates detailed captions using a Vision Model, and indexes them as searchable nodes.
4.  **Storage**:
    *   **Children/Images**: Stored in **ChromaDB** (Vector Store).
    *   **Parents**: Stored in a specialized **DocStore**.

### 2. Retrieval Pipeline
1.  **Query Analysis**: Detects if the user is asking for visual information (e.g., "Show me the fuse box").
2.  **Vector Search**: Finds relevant child chunks or image captions.
3.  **Reranking**: Uses a Cross-Encoder to score relevance.
4.  **Context Construction**: Retrieves the full "Parent" section for the top-ranked children.
5.  **Generation**: LLM answers the question using ONLY the provided context, with strict citation rules.

---

## 📂 Project Structure

The project has been refactored into a standard, modular backend structure:

```text
MechanicTroubleShooter/App/
├── api/                    #  API Endpoints
│   └── routes.py           # Routes for /chat, /ingest, /stats
├── core/                   #  Configuration
│   └── config.py           # Paths and settings
├── schemas/                #  Data Models
│   └── models.py           # Pydantic models (ChatRequest, etc.)
├── services/               #  Business Logic
│   ├── __init__.py         # Facade for easy imports
│   ├── ingestion/          # PDF Processing Pipeline
│   │   ├── pipeline.py     # Main Orchestrator
│   │   ├── chunking.py     # Parent/Child logic
│   │   ├── vision.py       # Image captioning
│   │   └── pdf_processor.py
│   ├── retrieval/          # Search Logic
│   │   ├── rag.py          # RAG Engine
│   │   └── reranker.py     # Cross-Encoder
│   ├── storage/            # Database Layer
│   │   ├── vector.py       # ChromaDB wrapper
│   │   └── document.py     # Parent Document Store
│   └── llm/                # AI Integration
│       ├── client.py       # Ollama Client
│       └── grader.py       # Hallucination Checker
└── main.py                 #  Application Entry Point
```

---

##  Getting Started

### Prerequisites

1.  **Python 3.10+**
2.  **Ollama** installed and running.
    *   Pull the required models:
        ```bash
        ollama pull llama3.2
        ollama pull llava-phi3  
        ```

### Installation

1.  Clone the repository and enter the directory.
2.  Create a virtual environment:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # Windows
    ```
3.  Install dependencies:
    ```bash
    pip install -r MechanicTroubleShooter/App/requirements.txt
    ```
    *(Note: Ensure you have `langchain`, `fastapi`, `chromadb`, `pymupdf`, `sentence-transformers` installed)*

### Running the Application

Navigate to the App directory and start the server:

```bash
cd MechanicTroubleShooter/App
uvicorn main:app --reload
```

The API will be available at: **http://localhost:8000**
Documentation (Swagger UI): **http://localhost:8000/docs**

---

##  API Usage

### 1. Ingest a Manual
Upload a PDF (e.g., Dacia Duster Manual) to index it.

*   **POST** `/ingest`
*   **Body**: `multipart/form-data`, file=`manual.pdf`

### 2. Chat
Ask questions about the ingested content.

*   **POST** `/chat`
*   **Body**:
    ```json
    {
      "query": "Where is the fuse for the radio located?",
      "k": 3
    }
    ```

### 3. Check Stats
View database status.

*   **GET** `/stats`

---

##  Technology Stack

*   **Backend**: FastAPI, Python
*   **LLM**: Ollama (Llama 3.2, LLaVA-Phi3)
*   **Vector DB**: ChromaDB
*   **Framework**: LangChain
*   **PDF Parsing**: PyMuPDF (Fitz)
