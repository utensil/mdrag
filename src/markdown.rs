use anyhow::Result;
use regex::Regex;

pub struct MarkdownParser;

impl MarkdownParser {
    pub fn parse(contents: &str, filename: &str) -> Result<Vec<String>> {
        // Find the first header type used in the document
        let header_regex = Regex::new(r"^(#{1,6})\s")?;

        let first_header_level = contents
            .lines()
            .find_map(|line| header_regex.captures(line).map(|caps| caps[1].to_string()));

        let chunks = match first_header_level {
            Some(header_prefix) => {
                // Create regex to split by this specific header level
                let pattern = format!(r"^{}\s", regex::escape(&header_prefix));
                let split_regex = Regex::new(&pattern)?;
                Self::split_by_header(contents, &split_regex, filename)?
            }
            None => {
                // No headers found, treat entire content as one chunk
                vec![format!("# {}\n{}", filename, contents)]
            }
        };

        // Further split chunks that are too large
        let final_chunks = chunks
            .into_iter()
            .flat_map(|chunk| Self::split_large_chunk(&chunk))
            .filter(|chunk| !chunk.trim().is_empty())
            .collect();

        Ok(final_chunks)
    }

    fn split_by_header(contents: &str, split_regex: &Regex, filename: &str) -> Result<Vec<String>> {
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut found_first_header = false;
        let mut pre_header_content = String::new();

        for line in contents.lines() {
            if split_regex.is_match(line) {
                // If we have content before the first header, save it
                if !found_first_header && !pre_header_content.trim().is_empty() {
                    chunks.push(format!("# {}\n{}", filename, pre_header_content.trim()));
                }

                // If we already found a header and have content, save the current chunk
                if found_first_header && !current_chunk.trim().is_empty() {
                    let chunk_content = current_chunk.trim();
                    chunks.push(format!("# {}\n{}", filename, chunk_content));
                    current_chunk = String::new();
                }
                found_first_header = true;
            }

            if found_first_header {
                current_chunk.push_str(line);
                current_chunk.push('\n');
            } else {
                pre_header_content.push_str(line);
                pre_header_content.push('\n');
            }
        }

        // Add the last chunk if it has content
        if !current_chunk.trim().is_empty() {
            let chunk_content = current_chunk.trim();
            chunks.push(format!("# {}\n{}", filename, chunk_content));
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
    fn test_parse_h2_headers() {
        let content = r#"
## foo
This is content under foo
Some more content

## bar
This is content under bar
More content here
"#;

        let chunks = MarkdownParser::parse(content, "test_file").unwrap();

        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].starts_with("# test_file\n## foo"));
        assert!(chunks[1].starts_with("# test_file\n## bar"));
        assert!(chunks[0].contains("This is content under foo"));
        assert!(chunks[1].contains("This is content under bar"));
    }

    #[test]
    fn test_parse_h1_with_mixed_headers() {
        let content = r#"
# foo
This is content under foo

### bar
This is a subsection

# foo2
This is content under foo2
"#;

        let chunks = MarkdownParser::parse(content, "test_file").unwrap();

        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].starts_with("# test_file\n# foo"));
        assert!(chunks[0].contains("### bar"));
        assert!(chunks[0].contains("This is content under foo"));
        assert!(chunks[1].starts_with("# test_file\n# foo2"));
        assert!(chunks[1].contains("This is content under foo2"));
        assert!(chunks[0].contains("### bar"));
        assert!(chunks[0].contains("This is a subsection"));
    }

    #[test]
    fn test_parse_no_headers() {
        let content = "This is just plain text without any headers.";

        let chunks = MarkdownParser::parse(content, "test_file").unwrap();

        assert_eq!(chunks.len(), 1);
        assert_eq!(
            chunks[0],
            "# test_file\nThis is just plain text without any headers."
        );
    }

    #[test]
    fn test_parse_does_not_start_with_header() {
        let content = r#"
This is just plain text without any headers.

# Section 1
This is content under section 1

# Section 2
This is content under section 2
"#;
        let chunks = MarkdownParser::parse(content, "test_file").unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(
            chunks[0],
            "# test_file\nThis is just plain text without any headers."
        );
        assert_eq!(
            chunks[1],
            "# test_file\n# Section 1\nThis is content under section 1"
        );
        assert_eq!(
            chunks[2],
            "# test_file\n# Section 2\nThis is content under section 2"
        );
    }

    #[test]
    fn test_parse_with_filename_all_chunks() {
        let content = r#"
## foo
This is content under foo
Some more content

## bar
This is content under bar
More content here
"#;

        let chunks = MarkdownParser::parse(content, "my_file").unwrap();

        assert_eq!(chunks.len(), 2);
        assert_eq!(
            chunks[0],
            "# my_file\n## foo\nThis is content under foo\nSome more content"
        );
        assert_eq!(
            chunks[1],
            "# my_file\n## bar\nThis is content under bar\nMore content here"
        );
    }

    #[test]
    fn test_large_chunk_splitting() {
        // Create a chunk larger than 8192 characters
        let large_content = "a".repeat(10000);
        let content = format!("## Large Section\n{}", large_content);

        let chunks = MarkdownParser::parse(&content, "test_file").unwrap();

        assert!(chunks.len() > 1);
        assert!(chunks[0].starts_with("# test_file\n## Large Section"));
        assert!(!chunks[1].starts_with("# test_file\n## Large Section"));

        // Each chunk should be <= 8192 characters
        for chunk in &chunks {
            assert!(chunk.len() <= 8192);
        }
    }

    #[test]
    fn test_chunk_splitting_respects_newlines() {
        // Create content with strategic newlines
        let content = format!(
            "## Test\n{}\n{}\n{}",
            "a".repeat(4000),
            "b".repeat(4000),
            "c".repeat(4000)
        );

        let chunks = MarkdownParser::parse(&content, "test_file").unwrap();

        // Should be split, and splits should happen at newlines when possible
        assert!(chunks.len() > 1);

        for chunk in &chunks {
            assert!(chunk.len() <= 8192);
        }
    }

    #[test]
    fn test_empty_content() {
        let chunks = MarkdownParser::parse("", "test_file").unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "# test_file\n");
    }

    #[test]
    fn test_whitespace_only_content() {
        let chunks = MarkdownParser::parse("   \n\n   ", "test_file").unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "# test_file\n   \n\n   ");
    }
}
