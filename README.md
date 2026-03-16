# JAN — Personal Knowledge Assistant

AI assistant that searches your **Obsidian vault** using RAG (Retrieval-Augmented Generation) with parent-child chunking, vector search, and conversational memory.

Built with **LangGraph** + **Qdrant** + **OpenRouter**.

## Quick Start

```bash
git clone https://github.com/<your-username>/Multi-Modal-Personal-Knowledge-Retriever.git
cd Multi-Modal-Personal-Knowledge-Retriever
./start.sh
```

That's it. The script will:

1. Create a Python virtual environment
2. Install all dependencies
3. Check if Qdrant is running (offers to start via Docker)
4. Walk you through an interactive setup if `.env` doesn't exist
5. Launch the assistant

## Prerequisites

- **Python 3.12+**
- **Docker** (for Qdrant vector database)
- **OpenRouter API key** — get one at [openrouter.ai/keys](https://openrouter.ai/keys)

## Manual Setup

If you prefer setting things up manually:

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -e .

# 3. Start Qdrant
docker run -d --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v ./data/qdrant_storage:/qdrant/storage \
  qdrant/qdrant

# 4. Copy and fill in your config
cp .env.example .env
# Edit .env with your API keys and vault path

# 5. Run
python -m src.main
```

## CLI Commands

| Command   | Description              |
|-----------|--------------------------|
| `/new`    | Start a new conversation |
| `/status` | Show vault index status  |
| `/quit`   | Exit the assistant       |

## Architecture

```
Obsidian Vault → Markdown Parser → Parent-Child Chunker
  → Embeddings (OpenRouter) → Qdrant (vector search)
  → SQLite (parent chunks + metadata)

User Query → LangGraph ReAct Agent → Tool: search_knowledge
  → Retrieve child chunks → Fetch parent context → LLM response
```

## Project Structure

```
src/
├── agents/          # LangGraph agent (graph, nodes, tools, prompts)
├── core/            # Config, data models
├── infrastructure/  # Qdrant, SQLite, Obsidian parser
├── services/        # Embedding, chunking, indexing, retrieval
└── main.py          # CLI entry point
```

## License

MIT
