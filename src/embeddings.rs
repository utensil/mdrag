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

    pub async fn embed_dir(&self, path: impl AsRef<Path>, model: &str) -> Result<()> {
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
        
        for file_path in files.iter() {
            pb.set_message(format!("{}", file_path.display()));
            self.embed_file(file_path, model).await?;
            pb.inc(1);
        }
        
        pb.finish_with_message("Completed");
        Ok(())
    }

    pub async fn embed_file(&self, path: impl AsRef<Path>, model: &str) -> Result<()> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let table_name = Self::get_table_name(model);

        // Check if file already has embeddings with this model
        let query = format!("SELECT COUNT(*) FROM {} WHERE path = ?", table_name);
        let count: Result<i64, _> = self.conn.query_row(&query, [&path_str], |row| row.get(0));
        
        // If table doesn't exist or file not embedded yet
        if count.unwrap_or(0) > 0 {
            debug!("File already embedded with model {}, skipping: {}", model, path_str);
            return Ok(());
        }

        debug!("Embedding file: {}", &path_str);

        let contents = fs::read_to_string(&path_str)?;
        let chunks = MarkdownParser::parse(&contents, &path_str)?;

        debug!("Split {} into {} chunks", &path_str, chunks.len());

        for (i, chunk) in chunks.iter().enumerate() {
            // Skip chunks that are still too large (safety check)
            if chunk.len() > 4000 {
                debug!("Skipping oversized chunk {} from {}: {} chars", i, &path_str, chunk.len());
                continue;
            }
            // Try to embed, but skip if it fails (chunk too dense)
            if let Err(e) = self.embed_chunk(&path_str, &chunk, model).await {
                debug!("Skipping chunk {} from {} due to error: {}", i, &path_str, e);
                continue;
            }
        }

        Ok(())
    }

    async fn embed_chunk(&self, file_path: &str, chunk: &str, model: &str) -> Result<()> {
        let embeddings = self.generate_embeddings(chunk, model).await?;
        
        // Ensure table exists with correct dimensions
        let dimension = embeddings[0].len();
        self.ensure_table_for_model(model, dimension)?;
        
        let table_name = Self::get_table_name(model);
        let query = format!(
            "INSERT INTO {} (path, contents, embedding) VALUES (?, ?, ?)",
            table_name
        );

        self.conn.execute(&query, (file_path, chunk, embeddings[0].as_bytes()))?;

        Ok(())
    }

    pub async fn search(&self, query: &str, k: usize, model: &str) -> Result<Vec<String>> {
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
            "SELECT contents
            FROM {}
            WHERE embedding MATCH ?1
            AND k = ?2
            ORDER BY distance",
            table_name
        );
        
        let mut stmt = self.conn.prepare(&sql)?;
        let results = stmt
            .query_map((query_embedding[0].as_bytes(), k as i32), |row| {
                Ok(row.get(0)?)
            })?
            .collect::<Result<Vec<String>, _>>()?;

        Ok(results)
    }
}
