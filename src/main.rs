use anyhow::Result;
use env_logger::Env;
use log::debug;
use ollama_rs::{Ollama, generation::completion::request::GenerationRequest, models::ModelOptions};
use rusqlite::{Connection, ffi::sqlite3_auto_extension};
use sqlite_vec::sqlite3_vec_init;
use tokio::io::{self, AsyncWriteExt};
use tokio_stream::StreamExt;

use crate::embeddings::Embedder;

mod embeddings;
mod markdown;

#[tokio::main]
async fn main() -> Result<()> {
    let env = Env::new().filter_or("RUST_LOG", "debug");
    env_logger::Builder::from_env(env).init();

    let args: Vec<String> = std::env::args().collect();
    let query = args
        .get(1)
        .expect("Please provide a query as the first argument");

    unsafe {
        sqlite3_auto_extension(Some(std::mem::transmute(sqlite3_vec_init as *const ())));
    }

    let db = Connection::open("markdown-rag.db")?;
    let ollama = Ollama::default();
    let embedder = Embedder::new(&db, &ollama)?;

    embedder.embed_dir("vault").await?;

    let results = embedder.search(&query, 5).await?;
    debug!("Results: {:?}", results);

    let model = "gemma3:1b".to_string();
    let options = ModelOptions::default()
        .temperature(0.2)
        .top_k(25)
        .top_p(0.25);
    let mut prompt = format!(
        "Answer the following user question: {query}\n\nUsing the following information for context:\n\n"
    );
    results
        .iter()
        .for_each(|result| prompt.push_str(&format!("\n\n{}", result)));

    debug!("Prompt: {}", prompt);

    let mut stream = ollama
        .generate_stream(GenerationRequest::new(model, prompt).options(options))
        .await?;

    let mut stdout = io::stdout();
    while let Some(res) = stream.next().await {
        let responses = res?;
        for resp in responses {
            stdout.write_all(resp.response.as_bytes()).await?;
            stdout.flush().await?;
        }
    }

    Ok(())
}
