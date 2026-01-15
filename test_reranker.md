# Testing Reranker Implementation

## Approach: LLM-based Scoring

Following community best practices for reranker models:

### Key Principles
1. **Minimal prompt** - No verbose instructions, just Query/Document/Score
2. **Temperature = 0** - Deterministic scoring for consistency
3. **Single document per call** - Cross-encoder pattern (one query-doc pair at a time)
4. **Clear output format** - Request score only (0-10)

### Implementation
```rust
// Minimal prompt format
let prompt = format!(
    "Query: {}\nDocument: {}\nScore (0-10):",
    query, document
);

// Temperature = 0 for deterministic scoring
let options = ModelOptions::default().temperature(0.0);
```

### Usage
```bash
# With reranking (default - uses dengcao/Qwen3-Reranker-0.6B:Q8_0)
cargo run --release -- search "your query"

# Without reranking (pure vector search)
cargo run --release -- search "your query" --reranker none

# With custom reranker model
cargo run --release -- search "your query" --reranker dengcao/Qwen3-Reranker-4B:Q5_K_M
```

### Expected Behavior
- Retrieves k*4 candidates (20 for k=5)
- Scores each with reranker model (0-10 scale)
- Sorts by score and returns top k (5)
- Should prioritize documents with exact keyword matches over semantically related but irrelevant content
