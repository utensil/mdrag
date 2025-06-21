use std::{fs, path::Path};

use anyhow::Result;
use glob::glob;
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
            embedding FLOAT[768]
        )",
            [],
        )?;

        Ok(Self { conn, ollama })
    }

    pub async fn generate_embeddings(&self, text: &str) -> Result<Vec<Vec<f32>>> {
        let request =
            GenerateEmbeddingsRequest::new("nomic-embed-text:v1.5".to_string(), text.into());
        let response = self.ollama.generate_embeddings(request).await?;

        Ok(response.embeddings)
    }

    pub async fn embed_dir(&self, path: impl AsRef<Path>) -> Result<()> {
        for entry in glob(&path.as_ref().join("**/*.md").to_string_lossy())? {
            let path = entry?;
            self.embed_file(&path).await?;
        }

        Ok(())
    }

    pub async fn embed_file(&self, path: impl AsRef<Path>) -> Result<()> {
        let path_str = path.as_ref().to_string_lossy().to_string();

        // Check if file already has embeddings
        let mut stmt = self
            .conn
            .prepare("SELECT COUNT(*) FROM file_embeddings WHERE path = ?")?;
        let count: i64 = stmt.query_row((&path_str,), |row| row.get(0))?;
        if count > 0 {
            debug!("File already embedded, skipping: {}", path_str);
            return Ok(());
        }

        debug!("Embedding file: {}", &path_str);

        let contents = fs::read_to_string(&path_str)?;
        let chunks = MarkdownParser::parse(&contents, &path_str)?;

        debug!("Split {} into {} chunks", &path_str, chunks.len());

        for chunk in chunks {
            self.embed_chunk(&path_str, &chunk).await?;
        }

        Ok(())
    }

    async fn embed_chunk(&self, file_path: &str, chunk: &str) -> Result<()> {
        let embeddings = self.generate_embeddings(chunk).await?;

        let mut stmt = self
            .conn
            .prepare("INSERT INTO file_embeddings (path, contents, embedding) VALUES (?, ?, ?)")?;

        stmt.execute((file_path, chunk, embeddings[0].as_bytes()))?;

        Ok(())
    }

    pub async fn search(&self, query: &str, k: usize) -> Result<Vec<String>> {
        let query_embedding = self.generate_embeddings(query).await?;

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
