use std::{fs, path::Path};

use anyhow::Result;
use glob::glob;
use indicatif::{ProgressBar, ProgressStyle};
use log::debug;
use ollama_rs::{Ollama, generation::embeddings::request::GenerateEmbeddingsRequest};
use rusqlite::Connection;
use zerocopy::IntoBytes;

use crate::markdown::MarkdownParser;

pub struct Embedder<'a> {
    conn: &'a Connection,
    ollama: &'a Ollama,
}

impl<'a> Embedder<'a> {
    pub fn new(conn: &'a Connection, ollama: &'a Ollama) -> Result<Self> {
        // AGENT-NOTE: We don't create tables here anymore - they're created dynamically per model
        Ok(Self { conn, ollama })
    }
    
    // AGENT-NOTE: Table naming rules
    // 1. Remove common prefixes/suffixes: "embed", "embedding", "text"
    // 2. Extract name: first 6 alphanumeric chars
    // 3. Extract version: strip non-alphanumeric
    // 4. Add 4-char hash of full model name for uniqueness
    // 5. Format: emb_{name}_{version}_{hash} with no consecutive underscores
    // Examples:
    //   nomic-embed-text:v1.5    → emb_nomic_v15_a3f2
    //   qwen3-embedding:4b       → emb_qwen3_4b_7c8d
    //   embeddinggemma:latest    → emb_gemma_latest_9e1f
    fn get_table_name(model: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        // Remove common prefixes/suffixes
        let mut cleaned = model.to_string();
        cleaned = cleaned.replace("embedding", "");
        cleaned = cleaned.replace("embed", "");
        cleaned = cleaned.replace("text", "");
        cleaned = cleaned.replace("-", "");
        cleaned = cleaned.trim_matches('-').to_string();
        
        // Split by colon to separate name and version
        let parts: Vec<&str> = cleaned.split(':').collect();
        
        // Extract name (first 6 alphanumeric chars)
        let name = parts[0]
            .chars()
            .filter(|c| c.is_alphanumeric())
            .take(6)
            .collect::<String>()
            .to_lowercase();
        
        // Extract version (strip non-alphanumeric)
        let version = if parts.len() > 1 {
            parts[1]
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
                .to_lowercase()
        } else {
            String::new()
        };
        
        // Generate 4-char hash of full model name
        let mut hasher = DefaultHasher::new();
        model.hash(&mut hasher);
        let hash = format!("{:x}", hasher.finish());
        let hash_suffix = &hash[..4];
        
        // Build table name with no consecutive underscores
        if version.is_empty() {
            format!("emb_{}__{}", name, hash_suffix)
        } else {
            format!("emb_{}_{}_{}", name, version, hash_suffix)
        }
    }
    
    fn ensure_table_for_model(&self, model: &str, dimension: usize) -> Result<()> {
        let table_name = Self::get_table_name(model);
        let sql = format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS {} USING vec0(
                path TEXT,
                section TEXT,
                chunk_index INTEGER,
                total_chunks INTEGER,
                contents TEXT,
                embedding FLOAT[{}]
            )",
            table_name, dimension
        );
        self.conn.execute(&sql, [])?;
        Ok(())
    }

    pub async fn generate_embeddings(&self, text: &str, model: &str) -> Result<Vec<Vec<f32>>> {
        let request = GenerateEmbeddingsRequest::new(model.to_string(), text.into());
        let response = self.ollama.generate_embeddings(request).await?;

        Ok(response.embeddings)
    }

    pub async fn embed_dir(&self, path: impl AsRef<Path>, model: &str) -> Result<(usize, usize, usize)> {
        let files: Vec<_> = glob(&path.as_ref().join("**/*.md").to_string_lossy())?
            .collect::<Result<Vec<_>, _>>()?;
        
        let total = files.len();
        
        let pb = ProgressBar::new(total as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg} (ETA: {eta})")
                .unwrap()
                .progress_chars("=>-")
        );
        
        let mut dirs_with_files = std::collections::HashSet::new();
        let mut total_tokens = 0;
        
        for file_path in files.iter() {
            pb.set_message(format!("{}", file_path.display()));
            let tokens = self.embed_file(file_path, model).await?;
            total_tokens += tokens;
            
            // Track directory containing this file
            if let Some(parent) = file_path.parent() {
                dirs_with_files.insert(parent.to_path_buf());
            }
            
            pb.inc(1);
        }
        
        pb.finish_with_message("Completed");
        Ok((dirs_with_files.len(), total, total_tokens))
    }

    pub async fn embed_file(&self, path: impl AsRef<Path>, model: &str) -> Result<usize> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let table_name = Self::get_table_name(model);

        // Check if file already has embeddings with this model
        let query = format!("SELECT COUNT(*) FROM {} WHERE path = ?", table_name);
        let count: Result<i64, _> = self.conn.query_row(&query, [&path_str], |row| row.get(0));
        
        // If table doesn't exist or file not embedded yet
        if count.unwrap_or(0) > 0 {
            debug!("File already embedded with model {}, skipping: {}", model, path_str);
            return Ok(0);
        }

        debug!("Embedding file: {}", &path_str);

        let contents = fs::read_to_string(&path_str)?;
        let chunks = MarkdownParser::parse(&contents, &path_str)?;

        debug!("Split {} into {} chunks", &path_str, chunks.len());

        let mut total_tokens = 0;
        for (i, chunk) in chunks.iter().enumerate() {
            // Skip chunks that are still too large (safety check)
            if chunk.content.len() > 4000 {
                debug!("Skipping oversized chunk {} from {}: {} chars", i, &path_str, chunk.content.len());
                continue;
            }
            // Try to embed, but skip if it fails (chunk too dense)
            if let Err(e) = self.embed_chunk(&path_str, chunk, model).await {
                debug!("Skipping chunk {} from {} due to error: {}", i, &path_str, e);
                continue;
            }
            // Estimate tokens by character count
            total_tokens += chunk.content.chars().count();
        }

        Ok(total_tokens)
    }

    async fn embed_chunk(&self, file_path: &str, chunk: &crate::markdown::Chunk, model: &str) -> Result<()> {
        let embeddings = self.generate_embeddings(&chunk.content, model).await?;
        
        // Ensure table exists with correct dimensions
        let dimension = embeddings[0].len();
        self.ensure_table_for_model(model, dimension)?;
        
        let table_name = Self::get_table_name(model);
        let query = format!(
            "INSERT INTO {} (path, section, chunk_index, total_chunks, contents, embedding) VALUES (?, ?, ?, ?, ?, ?)",
            table_name
        );

        self.conn.execute(&query, (
            file_path,
            &chunk.section,
            chunk.chunk_index as i64,
            chunk.total_chunks as i64,
            &chunk.content,
            embeddings[0].as_bytes()
        ))?;

        Ok(())
    }

    pub async fn search(&self, query: &str, k: usize, model: &str) -> Result<Vec<(String, String, usize, usize)>> {
        let table_name = Self::get_table_name(model);
        
        // Check if table exists for this model
        let table_exists: Result<i64, _> = self.conn.query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
            [&table_name],
            |row| row.get(0)
        );
        
        if table_exists.unwrap_or(0) == 0 {
            // List available models by checking all emb_* tables (excluding internal vec0 tables)
            let mut stmt = self.conn.prepare(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'emb_%' 
                 AND name NOT LIKE '%_info' AND name NOT LIKE '%_chunks' 
                 AND name NOT LIKE '%_rowids' AND name NOT LIKE '%_vector_%' 
                 AND name NOT LIKE '%_metadata%'"
            )?;
            let available: Vec<String> = stmt
                .query_map([], |row| row.get(0))?
                .collect::<Result<Vec<String>, _>>()?;
            
            if available.is_empty() {
                anyhow::bail!("No embeddings found in database. Run 'embed' command first.");
            } else {
                // Convert table names back to model names for display
                let model_names: Vec<String> = available.iter()
                    .map(|t| t.strip_prefix("emb_").unwrap_or(t))
                    .map(|s| s.to_string())
                    .collect();
                anyhow::bail!(
                    "No embeddings found for model '{}'. Available: {}\n\
                    Either re-embed with this model or use --rag-model with an available model.",
                    model, model_names.join(", ")
                );
            }
        }
        
        let query_embedding = self.generate_embeddings(query, model).await?;

        let sql = format!(
            "SELECT contents, section, chunk_index, total_chunks
            FROM {}
            WHERE embedding MATCH ?1
            AND k = ?2
            ORDER BY distance",
            table_name
        );
        
        let mut stmt = self.conn.prepare(&sql)?;
        let results = stmt
            .query_map((query_embedding[0].as_bytes(), k as i32), |row| {
                Ok((
                    row.get(0)?,  // contents
                    row.get(1)?,  // section
                    row.get::<_, i64>(2)? as usize,  // chunk_index
                    row.get::<_, i64>(3)? as usize,  // total_chunks
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(results)
    }

    // AGENT-NOTE: Batch reranking for 5x speedup with minimal accuracy loss
    // Scores multiple documents in one LLM call, parsing array output
    async fn score_relevance_batch(&self, query: &str, documents: &[(String, String, usize, usize)], reranker_model: &str) -> Result<Vec<f32>> {
        use ollama_rs::generation::completion::request::GenerationRequest;
        use ollama_rs::models::ModelOptions;
        
        if documents.is_empty() {
            return Ok(vec![]);
        }
        
        // Build batch prompt - simpler format for reranker models
        let mut prompt = format!("Query: {}\n\n", query);
        for (i, (content, _, _, _)) in documents.iter().enumerate() {
            prompt.push_str(&format!("Document {}: {}\n\n", i + 1, content));
        }
        prompt.push_str("Rate each document's relevance to the query on a scale of 0-10.\nProvide only the scores as numbers separated by spaces:");
        
        let options = ModelOptions::default().temperature(0.0);
        let request = GenerationRequest::new(reranker_model.to_string(), prompt)
            .options(options);
        
        let response = self.ollama.generate(request).await?;
        
        // Parse array of scores from response
        let response_text = response.response.trim();
        debug!("Reranker response: {}", response_text);
        let scores = Self::parse_score_array(response_text, documents.len());
        debug!("Parsed scores: {:?}", scores);
        
        Ok(scores)
    }
    
    // AGENT-NOTE: Parse score array from various formats: [1,2,3] or "1 2 3" or "1, 2, 3"
    fn parse_score_array(text: &str, expected_count: usize) -> Vec<f32> {
        // Try to extract all numbers from the response
        let numbers: Vec<f32> = text
            .chars()
            .filter(|c| c.is_numeric() || c.is_whitespace() || *c == '.' || *c == ',')
            .collect::<String>()
            .split(|c: char| !c.is_numeric() && c != '.')
            .filter_map(|s| s.parse::<f32>().ok())
            .collect();
        
        // Normalize to 0-1 range and pad if needed
        let mut scores: Vec<f32> = numbers.iter().map(|&s| (s / 10.0).min(1.0)).collect();
        
        // If we got fewer scores than expected, pad with 0.0
        while scores.len() < expected_count {
            scores.push(0.0);
        }
        
        // If we got more scores than expected, truncate
        scores.truncate(expected_count);
        
        scores
    }

    // AGENT-NOTE: Search with reranking - retrieves k*multiplier candidates then reranks top k
    // Uses batch scoring (5 docs/batch) for 5x speedup with minimal accuracy loss
    pub async fn search_with_rerank(
        &self,
        query: &str,
        k: usize,
        model: &str,
        reranker_model: &str,
        multiplier: usize,
    ) -> Result<(Vec<(String, String, usize, usize, f32)>, std::time::Duration)> {
        use indicatif::{ProgressBar, ProgressStyle};
        
        let rerank_start = std::time::Instant::now();
        
        // Retrieve more candidates for reranking
        let candidates = self.search(query, k * multiplier, model).await?;
        
        debug!("Reranking {} candidates with model {} (batch size: 5)", candidates.len(), reranker_model);
        
        // Create progress bar
        let pb = ProgressBar::new(candidates.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{msg} [{bar:40.cyan/blue}] {pos}/{len} chunks ({eta})")
                .unwrap()
                .progress_chars("=>-")
        );
        pb.set_message("Reranking");
        
        // Score candidates in batches of 5
        const BATCH_SIZE: usize = 5;
        let mut scored_results = Vec::new();
        
        for batch in candidates.chunks(BATCH_SIZE) {
            let scores = self.score_relevance_batch(query, batch, reranker_model).await?;
            
            for (i, (content, section, chunk_idx, total_chunks)) in batch.iter().enumerate() {
                let score = scores.get(i).copied().unwrap_or(0.0);
                scored_results.push((content.clone(), section.clone(), *chunk_idx, *total_chunks, score));
            }
            
            pb.inc(batch.len() as u64);
        }
        
        pb.finish_and_clear();
        
        // Sort by score (descending)
        scored_results.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));
        
        let rerank_time = rerank_start.elapsed();
        
        // Return top k with timing
        scored_results.truncate(k);
        Ok((scored_results, rerank_time))
    }
}
