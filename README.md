# HR Policy Assistant

An AI-powered HR Policy Assistant built with RAG (Retrieval-Augmented Generation) using NVIDIA's AI endpoints, LangChain, and Streamlit.

## Overview

This application allows employees to ask questions about HR policies and receive accurate answers based on official company policy documents. The system uses RAG with NVIDIA embeddings and reranking for accurate document retrieval, and NVIDIA's LLaMA 3.1 8B model for generation.


### Components

| Component | Technology |
|-----------|------------|
| Embeddings | NVIDIA `nvidia/nv-embed-v1` |
| LLM | NVIDIA `meta/llama-3.1-8b-instruct` |
| Reranker | NVIDIA `nv-rerank-qa-mistral-4b:1` |
| Vector Store | ChromaDB via LangChain |
| Text Splitter | LangChain Text Splitters |
| Frontend | Streamlit |
| Evaluation | RAGAS |


## Prerequisites

- Python 3.10+
- NVIDIA API key (set in `.env` file)
- ChromaDB (installed via `langchain-chroma`)

## Installation

```bash
# Clone the repository
cd HR-Policy-Assist

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file with your NVIDIA API key
echo "NVIDIA_API_KEY=your_api_key_here" > .env
```

## Usage

### 1. Ingest Documents (Run Once)

```bash
python main.py
```

This will:
1. Load documents from `./Content`
2. Split into chunks
3. Generate embeddings using NVIDIA embeddings
4. Store in ChromaDB at `./HR-Policy-Chunks`
5. Run RAGAS evaluation

### 2. Run the Web App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### 3. Run Evaluation

```bash
python -m Evaluation.run_evaluation
```

Runs RAGAS evaluation on the RAG pipeline.

## Configuration

Edit `consts.py` to configure:

```python
class Consts:
    def __init__(self):
        self.folder_path = './Content'           # Document source
        self.persist_directory = './HR-Policy-Chunks'  # Vector DB path
        self.embegging = NVIDIAEmbeddings(model='nvidia/nv-embed-v1')
        self.llm = ChatNVIDIA(
            model="meta/llama-3.1-8b-instruct",
            temperature=0
        )
```

## Environment Variables

Create a `.env` file:

```env
NVIDIA_API_KEY=your_nvidia_api_key
```

## Project Details

### Content Structure

The `Content/` directory contains HR policies organized by category:
- **Benefits & Perks**: Parental leave, nomad life, mental health, spending
- **Compensation**: Total rewards, leveling, equity, compensation changes
- **Policies**: Security, access control, data management, business continuity
- **Company Info**: Values, communication, community, code of conduct
- **Departments**: People & Talent processes

### RAG Pipeline

1. **Ingestion**: Loads Markdown files from `Content/`
2. **Chunking**: Splits documents using LangChain text splitters
3. **Embedding**: NVIDIA `nv-embed-v1` embeddings
4. **Storage**: ChromaDB persistent vector store
5. **Retrieval**: Similarity search (k=30) → NVIDIA reranker (top-5)
6. **Generation**: NVIDIA LLaMA 3.1 8B with strict context-only prompt
7. **Evaluation**: RAGAS metrics (faithfulness, answer_relevancy, etc.)

### Streamlit App Features

- Chat interface for HR policy questions
- Conversation history in session
- Real-time retrieval with spinner feedback
- Strict context-only responses (no hallucination)
- Professional, concise answers

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI |
| `langchain-core` | Core LangChain abstractions |
| `langchain-nvidia-ai-endpoints` | NVIDIA embeddings, LLM, reranker |
| `langchain-chroma` | ChromaDB integration |
| `langchain-text-splitters` | Document chunking |
| `langchain-community` | Community integrations |
| `langchain-classic` | Contextual compression retriever |
| `ragas` | RAG evaluation |
| `python-dotenv` | Environment variables |

## Evaluation

Run RAGAS evaluation to measure:
- **Faithfulness**: Answer faithfulness to context
- **Answer Relevancy**: Answer relevance to question
- **Context Precision**: Relevant context retrieval
- **Context Recall**: Complete context retrieval

```bash
python -m Evaluation.run_evaluation
```

## Development

```bash
# Run linting (if configured)
pip install ruff
ruff check .

# Type checking (if configured)
pip install mypy
mypy .
```