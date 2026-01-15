use anyhow::Result;
use regex::Regex;

#[derive(Debug, Clone)]
pub struct Chunk {
    pub content: String,
    pub section: String,
    pub chunk_index: usize,
    pub total_chunks: usize,
}

pub struct MarkdownParser;

impl MarkdownParser {
    pub fn parse(contents: &str, filename: &str) -> Result<Vec<Chunk>> {
        // Find the first header type used in the document
        let header_regex = Regex::new(r"^(#{1,6})\s")?;

        let first_header_level = contents
            .lines()
            .find_map(|line| header_regex.captures(line).map(|caps| caps[1].to_string()));

        let section_chunks = match first_header_level {
            Some(header_prefix) => {
                // Create regex to split by this specific header level
                let pattern = format!(r"^{}\s", regex::escape(&header_prefix));
                let split_regex = Regex::new(&pattern)?;
                Self::split_by_header(contents, &split_regex, filename)?
            }
            None => {
                // No headers found, treat entire content as one chunk
                vec![(filename.to_string(), format!("# {}\n{}", filename, contents))]
            }
        };

        // Further split chunks that are too large and track positions
        let mut final_chunks = Vec::new();
        for (section, content) in section_chunks {
            let split_contents = Self::split_large_chunk(&content);
            let total = split_contents.len();
            for (idx, chunk_content) in split_contents.into_iter().enumerate() {
                if !chunk_content.trim().is_empty() {
                    final_chunks.push(Chunk {
                        content: chunk_content,
                        section: section.clone(),
                        chunk_index: idx + 1,
                        total_chunks: total,
                    });
                }
            }
        }

        Ok(final_chunks)
    }

    fn split_by_header(contents: &str, split_regex: &Regex, filename: &str) -> Result<Vec<(String, String)>> {
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut current_section = String::new();
        let mut found_first_header = false;
        let mut pre_header_content = String::new();

        for line in contents.lines() {
            if split_regex.is_match(line) {
                // If we have content before the first header, save it
                if !found_first_header && !pre_header_content.trim().is_empty() {
                    chunks.push((filename.to_string(), format!("# {}\n{}", filename, pre_header_content.trim())));
                }

                // Save the previous chunk if it has content
                if found_first_header && !current_chunk.trim().is_empty() {
                    let chunk_content = current_chunk.trim();
                    chunks.push((current_section.clone(), format!("# {}\n{}", filename, chunk_content)));
                }

                // Start a new chunk
                current_section = line.to_string();
                current_chunk = String::from(line);
                current_chunk.push('\n');
                found_first_header = true;
            } else {
                // Add line to current chunk or pre-header content
                if found_first_header {
                    current_chunk.push_str(line);
                    current_chunk.push('\n');
                } else {
                    pre_header_content.push_str(line);
                    pre_header_content.push('\n');
                }
            }
        }

        // Add the last chunk if it has content
        if !current_chunk.trim().is_empty() {
            let chunk_content = current_chunk.trim();
            chunks.push((current_section, format!("# {}\n{}", filename, chunk_content)));
        }

        Ok(chunks)
    }

    fn split_large_chunk(chunk: &str) -> Vec<String> {
        // AGENT-NOTE: Keep 8192 base size, filter base64, dynamically adjust only when needed
        const MAX_CHUNK_SIZE: usize = 8192;
        const MAX_TOKEN_ESTIMATE: usize = 4000; // ~2 chars per token for dense content

        // Remove base64 image data (useless for RAG)
        let base64_pattern = Regex::new(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]{100,}").unwrap();
        let cleaned = base64_pattern.replace_all(chunk, "[image removed]");

        if cleaned.len() <= MAX_CHUNK_SIZE {
            return vec![cleaned.to_string()];
        }

        let mut chunks = Vec::new();
        let mut start = 0;

        while start < cleaned.len() {
            // Try full size first
            let mut end = (start + MAX_CHUNK_SIZE).min(cleaned.len());
            
            // If this would create a chunk > token limit, shrink it
            if end - start > MAX_TOKEN_ESTIMATE {
                end = start + MAX_TOKEN_ESTIMATE;
            }

            // Ensure we're at a char boundary
            while end < cleaned.len() && !cleaned.is_char_boundary(end) {
                end -= 1;
            }

            let actual_end = if end < cleaned.len() {
                cleaned[start..end]
                    .rfind('\n')
                    .map(|pos| start + pos + 1)
                    .unwrap_or(end)
            } else {
                end
            };

            let chunk_text = &cleaned[start..actual_end];
            if !chunk_text.trim().is_empty() {
                chunks.push(chunk_text.to_string());
            }
            start = actual_end;
        }

        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_with_headers() {
        let content = "# Header 1\nContent 1\n# Header 2\nContent 2";
        let chunks = MarkdownParser::parse(content, "test_file").unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].section, "# Header 1");
        assert_eq!(chunks[0].chunk_index, 1);
        assert_eq!(chunks[0].total_chunks, 1);
    }

    #[test]
    fn test_parse_without_headers() {
        let content = "Just some content without headers";
        let chunks = MarkdownParser::parse(content, "test_file").unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].section, "test_file");
    }

    #[test]
    fn test_large_chunk_splitting() {
        let content = format!("# Header\n{}", "a".repeat(10000));
        let chunks = MarkdownParser::parse(&content, "test_file").unwrap();
        assert!(chunks.len() > 1);
        assert_eq!(chunks[0].section, "# Header");
        assert_eq!(chunks[1].section, "# Header"); // Same section
        assert_eq!(chunks[0].chunk_index, 1);
        assert_eq!(chunks[1].chunk_index, 2);
        assert_eq!(chunks[0].total_chunks, chunks[1].total_chunks);
    }
}
