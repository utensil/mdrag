use anyhow::Result;
use clap::{Parser, Subcommand};
use env_logger::Env;
use log::{debug, info};
use ollama_rs::{Ollama, generation::completion::request::GenerationRequest, models::ModelOptions};
use rusqlite::{Connection, ffi::sqlite3_auto_extension};
use sqlite_vec::sqlite3_vec_init;
use tokio::io::{self, AsyncWriteExt};
use tokio_stream::StreamExt;

use crate::embeddings::Embedder;

mod embeddings;
mod markdown;

#[derive(Parser)]
#[command(name = "mdrag")]
#[command(about = "RAG Pipeline for Markdown Files")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Embed {
        #[arg(value_name = "VAULT_PATH")]
        vault_path: String,
    },
    Search {
        #[arg(value_name = "QUERY")]
        query: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let env = Env::new().filter_or("RUST_LOG", "info");
    env_logger::Builder::from_env(env).init();

    let cli = Cli::parse();

    unsafe {
        sqlite3_auto_extension(Some(std::mem::transmute(sqlite3_vec_init as *const ())));
    }

    let db = Connection::open("mdrag.db")?;
    let ollama = Ollama::default();
    let embedder = Embedder::new(&db, &ollama)?;

    match &cli.command {
        Commands::Embed { vault_path } => {
            info!("Embedding files from directory: {}", vault_path);
            embedder.embed_dir(vault_path).await?;
            info!("Embedding completed successfully!");
        }
        Commands::Search { query } => {
            info!("Searching for: {}", query);

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
        }
    }

    Ok(())
}
