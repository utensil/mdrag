use anyhow::{Result, bail};
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

const SUPPORTED_EMBED_MODELS: &[&str] = &[
    "qwen3-embedding:4b",
    "qwen3-embedding:8b",
    "nomic-embed-text:v1.5",
    "embeddinggemma:latest",
];

const SUPPORTED_LLM_MODELS: &[&str] = &[
    "qwen3:14b",
    "qwen3:8b",
    "gemma3:1b",
];

async fn check_model_available(ollama: &Ollama, model: &str) -> Result<bool> {
    match ollama.list_local_models().await {
        Ok(models) => Ok(models.iter().any(|m| m.name == model)),
        Err(_) => Ok(false),
    }
}

async fn prompt_model_selection(ollama: &Ollama, model_type: &str, models: &[&str]) -> Result<String> {
    println!("\n{} model '{}' not found.", model_type, models[0]);
    println!("\nSupported {} models:", model_type);
    
    let mut has_available = false;
    for model in models.iter() {
        let available = check_model_available(ollama, model).await?;
        let status = if available { "✓" } else { "⬇" };
        let note = if available { "(available)" } else { "(needs download)" };
        println!("  {} {} {}", status, model, note);
        if available {
            has_available = true;
        }
    }
    
    println!("\nTo download a model, run:");
    println!("  ollama pull <model-name>");
    
    if has_available {
        println!("\nOr specify an available model with --model flag");
    }
    
    bail!("Model not available. Please download or specify an existing model.");
}

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
        #[arg(long)]
        model: Option<String>,
    },
    Search {
        #[arg(value_name = "QUERY")]
        query: String,
        #[arg(long)]
        rag_model: Option<String>,
        #[arg(long)]
        chat_model: Option<String>,
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
        Commands::Embed { vault_path, model } => {
            let embed_model = match model {
                Some(m) => m.clone(),
                None => {
                    if !check_model_available(&ollama, SUPPORTED_EMBED_MODELS[0]).await? {
                        prompt_model_selection(&ollama, "Embedding", SUPPORTED_EMBED_MODELS).await?
                    } else {
                        SUPPORTED_EMBED_MODELS[0].to_string()
                    }
                }
            };
            
            info!("Embedding files from directory: {} with model: {}", vault_path, embed_model);
            let start_time = std::time::Instant::now();
            let (dirs, files, tokens) = embedder.embed_dir(vault_path, &embed_model).await?;
            let total_time = start_time.elapsed();
            
            info!("Embedding completed successfully!");
            
            // Print stats in grey
            let grey = "\x1b[90m";
            let reset = "\x1b[0m";
            println!("\n{}{} dirs · {} files · {} tokens · embedded in {:.2}s{}\n", 
                grey, dirs, files, tokens, total_time.as_secs_f64(), reset);
        }
        Commands::Search { query, rag_model, chat_model } => {
            let embed = match rag_model {
                Some(m) => m.clone(),
                None => {
                    if !check_model_available(&ollama, SUPPORTED_EMBED_MODELS[0]).await? {
                        prompt_model_selection(&ollama, "Embedding", SUPPORTED_EMBED_MODELS).await?
                    } else {
                        SUPPORTED_EMBED_MODELS[0].to_string()
                    }
                }
            };
            
            let llm = match chat_model {
                Some(m) => m.clone(),
                None => {
                    if !check_model_available(&ollama, SUPPORTED_LLM_MODELS[0]).await? {
                        prompt_model_selection(&ollama, "LLM", SUPPORTED_LLM_MODELS).await?
                    } else {
                        SUPPORTED_LLM_MODELS[0].to_string()
                    }
                }
            };
            
            info!("Searching for: {} with embedding model: {}, LLM model: {}", query, embed, llm);

            let search_start = std::time::Instant::now();
            let results = embedder.search(&query, 5, &embed).await?;
            let search_time = search_start.elapsed();
            let num_chunks = results.len();
            debug!("Results: {:?}", results);
            debug!("Search took: {:.3}s", search_time.as_secs_f64());

            let options = ModelOptions::default()
                .temperature(0.2)
                .top_k(25)
                .top_p(0.25);
            let mut prompt = format!(
                "Answer the following user question: {query}\n\nUsing the following information for context:\n\n"
            );
            results
                .iter()
                .for_each(|(content, section, chunk_idx, total_chunks)| {
                    let chunk_info = if *total_chunks > 1 {
                        format!(" (chunk {}/{})", chunk_idx, total_chunks)
                    } else {
                        String::new()
                    };
                    prompt.push_str(&format!("\n\n[Section: {}{}]\n{}", section, chunk_info, content));
                });

            debug!("Prompt: {}", prompt);

            let generation_start = std::time::Instant::now();
            let mut stream = ollama
                .generate_stream(GenerationRequest::new(llm, prompt).options(options))
                .await?;

            let mut stdout = io::stdout();
            let mut first_token_time: Option<std::time::Duration> = None;
            let mut token_count = 0;
            
            while let Some(res) = stream.next().await {
                let responses = res?;
                for resp in responses {
                    if first_token_time.is_none() && !resp.response.is_empty() {
                        first_token_time = Some(generation_start.elapsed());
                    }
                    
                    stdout.write_all(resp.response.as_bytes()).await?;
                    stdout.flush().await?;
                    
                    // Estimate tokens by character count
                    token_count += resp.response.chars().count();
                }
            }
            
            let generation_time = generation_start.elapsed();
            let total_time = search_time + generation_time;
            
            // Calculate reply time (from first token to end)
            let reply_time = if let Some(ttft) = first_token_time {
                generation_time - ttft
            } else {
                generation_time
            };
            
            // Estimate tokens (rough: 1 char ≈ 1 token for mixed content)
            let estimated_tokens = token_count;
            
            // Print stats in grey with compact format
            let grey = "\x1b[90m";
            let reset = "\x1b[0m";
            
            print!("\n\n{}", grey);
            print!("{:.2}s", total_time.as_secs_f64());
            print!(" · searched for {:.2}s", search_time.as_secs_f64());
            print!(" · found {} chunks", num_chunks);
            if let Some(ttft) = first_token_time {
                print!(" · thought for {:.2}s", ttft.as_secs_f64());
            }
            print!(" · replied in {:.2}s", reply_time.as_secs_f64());
            if estimated_tokens > 0 && reply_time.as_secs_f64() > 0.0 {
                print!(" · {:.0} tok/s", estimated_tokens as f64 / reply_time.as_secs_f64());
            }
            println!("{}\n", reset);
        }
    }

    Ok(())
}
