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
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS file_embeddings USING vec0(
            path TEXT,
            contents TEXT,
            model TEXT,
            embedding FLOAT[768]
        )",
            [],
        )?;

        Ok(Self { conn, ollama })
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

        // Check if file already has embeddings with this model
        let mut stmt = self
            .conn
            .prepare("SELECT COUNT(*) FROM file_embeddings WHERE path = ? AND model = ?")?;
        let count: i64 = stmt.query_row((&path_str, model), |row| row.get(0))?;
        if count > 0 {
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

        let mut stmt = self
            .conn
            .prepare("INSERT INTO file_embeddings (path, contents, model, embedding) VALUES (?, ?, ?, ?)")?;

        stmt.execute((file_path, chunk, model, embeddings[0].as_bytes()))?;

        Ok(())
    }

    pub async fn search(&self, query: &str, k: usize, model: &str) -> Result<Vec<String>> {
        let query_embedding = self.generate_embeddings(query, model).await?;

        let mut stmt = self.conn.prepare(
            "SELECT contents
            FROM file_embeddings
            WHERE embedding MATCH ?1
            AND k = ?2
            ORDER BY distance",
        )?;

        let results = stmt
            .query_map((query_embedding[0].as_bytes(), k as i32), |row| {
                Ok(row.get(0)?)
            })?
            .collect::<Result<Vec<String>, _>>()?;

        Ok(results)
    }
}
