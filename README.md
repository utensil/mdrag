# mdrag

A local RAG (Retrieval Augmented Generation) system for chatting with your Obsidian vault using Rust and local AI models.

## Features

- **Local-first**: Runs entirely offline, keeping your notes private
- **Intelligent chunking**: Splits markdown files by headers to maintain context
- **Vector search**: Uses SQLite with sqlite-vec extension for fast similarity search
- **Streaming responses**: Real-time AI-generated answers using local Ollama models

## Prerequisites

1. **Rust**
2. **Ollama**
3. **Required models**:
   ```bash
   ollama pull nomic-embed-text-v1.5
   ollama pull gemma3:1b
   ```

## Installation

```bash
git clone https://github.com/orellazri/mdrag.git
cd mdrag
cargo build --release
```

## Usage

### 1. Create embeddings from your vault

```bash
./target/release/mdrag embed /path/to/your/obsidian/vault
```

### 2. Search and chat with your notes

```bash
./target/release/mdrag search "How did I fix that Git submodule issue?"
```

## Example

```bash
# Process your markdown files
mdrag embed ~/Documents/MyVault

# Ask natural language questions
mdrag search "What were the main challenges in the refactor?"
mdrag search "Show me my Docker troubleshooting notes"
```

The system will find relevant chunks from your notes and generate conversational answers using the local AI model.

## How it works

1. **Embedding phase**: Parses markdown files, chunks them by headers, and generates vector embeddings
2. **Query phase**: Embeds your question, finds similar content, and uses a local LLM to synthesize an answer

All data stays on your machine - no cloud services required.
