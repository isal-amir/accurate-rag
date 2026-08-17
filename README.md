# Accurate RAG Chatbot (Support Assistant)

A highly advanced Retrieval-Augmented Generation (RAG) chatbot built to assist users with the "Accurate" online accounting software. This project employs a multi-modal PDF parser and a highly token-efficient memory management system to deliver fast, accurate, and context-aware responses.

## ✨ Key Features

- **Multimodal PDF Parsing**: Uses `gemini-3.5-flash-lite` to parse PDFs page-by-page. It can "see" images and flawlessly extract complex markdown tables, allowing it to index software screenshots and manuals effectively.
- **Active RAG (Self-Reflection)**: Implements LangGraph to build a dynamic routing pipeline. If a user asks a vague question and the initial document retrieval fails, the agent automatically rewrites the query using an LLM pass and tries again before falling back.
- **Token-Efficient Memory**: Avoids the trap of passing massive chat histories to the LLM. It tracks unsummarized messages and triggers a background batch-summarization of older turns once a threshold is reached. The main Generator LLM receives a strict limit of the 2 most recent chat turns plus a running summary.
- **Page-Level Citations**: Because chunking is done strictly within page boundaries, the chatbot accurately cites the exact page number the information was sourced from.

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Agent Orchestration**: LangChain & LangGraph
- **Vector Database**: Qdrant (Local via Docker)
- **Document Parsing**: PyMuPDF + Google Gemini (`gemini-3.5-flash-lite`)
- **Embeddings**: Google Generative AI (`gemini-embedding-2` - 3072 dims)
- **Generator Model**: OpenRouter (`poolside/laguna-s-2.1:free`)
- **Observability**: LangSmith

## 🚀 Setup & Installation

### 1. Prerequisites
Ensure you have Python, Conda, and Docker installed on your machine.

### 2. Install Dependencies
Activate your conda environment and install the required Python packages:
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory and configure your API keys:
```dotenv
OPENROUTER_API_KEY="your_openrouter_api_key"
GEMINI_API_KEY="your_gemini_api_key"

# LangSmith Configuration (Optional but recommended)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="your_langsmith_api_key"
LANGCHAIN_PROJECT="accurate_rag_project"
```

### 4. Start the Vector Database
Launch the local Qdrant container:
```bash
docker-compose up -d
```

### 5. Run the Application
Start the Streamlit interface:
```bash
streamlit run app.py
```

## 📖 Known Limitations
- Multi-page tables may be cut in half due to the strict page-by-page parsing approach. This is an intentional trade-off to ensure 100% accurate page-level citations for cross-checking facts.
