# mdrag

## Fork Enhancements

This fork adds significant improvements for production use with CJK (Chinese/Japanese/Korean) content and multi-model workflows:

### Key Features Added

**🚀 Multi-Model Support**
- Support multiple embedding models simultaneously (qwen3-embedding, nomic-embed-text, etc.)
- Separate tables per model with automatic dimension handling
- Smart model switching with `--rag-model` and `--chat-model` flags
- Model availability checking with helpful prompts

**📊 Rich Statistics**
- Embed stats: `1 dirs · 4 files · 10929 tokens · embedded in 11.98s`
- Search stats: `596 tok/sec · 3954 tokens · 0.00s to first token · 6.64s in total`
- Visual progress bar with ETA during embedding

**🌏 CJK Optimization**
- UTF-8 char boundary safety for multi-byte characters
- Smart chunk sizing with base64 image filtering
- Default models optimized for CJK: qwen3-embedding:4b + qwen3:14b

**🔍 Enhanced Metadata**
- Section tracking with chunk positions: `[Section: ## Introduction (chunk 1/3)]`
- Searchable by section names
- Clear context for LLM without confusion

Forked from [orellazri/mdrag](https://github.com/orellazri/mdrag). See [comparison with upstream](https://github.com/orellazri/mdrag/compare/main...utensil:mdrag:main) for detailed changes.

------

A local RAG (Retrieval Augmented Generation) system for chatting with your Obsidian vault using Rust and local AI models.

Original project by [orellazri](https://github.com/orellazri/mdrag). Read more in the [blog post](https://orellazri.com/posts/rag-pipeline-chat-with-my-obsidian-vault/).

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
