use crate::{Language, RecursiveCharacterTextSplitter, TextSplitter};
use langchain_core::documents::Document;
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeaderType {
    pub level: usize,
    pub name: String,
    pub data: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineType {
    pub content: String,
    pub metadata: BTreeMap<String, Value>,
}

#[derive(Debug, Clone)]
pub struct MarkdownTextSplitter {
    inner: RecursiveCharacterTextSplitter,
}

impl MarkdownTextSplitter {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            inner: RecursiveCharacterTextSplitter::from_language(
                Language::Markdown,
                chunk_size,
                chunk_overlap,
            ),
        }
    }
}

impl TextSplitter for MarkdownTextSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        self.inner.split_text(text)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MarkdownHeaderTextSplitter {
    headers_to_split_on: Vec<(String, String)>,
    return_each_line: bool,
    strip_headers: bool,
}

impl MarkdownHeaderTextSplitter {
    pub fn new(
        headers_to_split_on: Vec<(String, String)>,
        return_each_line: bool,
        strip_headers: bool,
    ) -> Self {
        let mut headers_to_split_on = headers_to_split_on;
        headers_to_split_on.sort_by(|left, right| right.0.len().cmp(&left.0.len()));
        Self {
            headers_to_split_on,
            return_each_line,
            strip_headers,
        }
    }

    pub fn split_text(&self, text: &str) -> Vec<Document> {
        let mut documents = Vec::new();
        let mut header_stack = Vec::<HeaderType>::new();
        let mut metadata = BTreeMap::new();
        let mut current_content = Vec::<String>::new();
        let mut in_code_block = false;
        let mut fence = "";

        for line in text.lines() {
            let trimmed = line.trim();
            if is_fence_line(trimmed, "```") {
                in_code_block = !matches!(fence, "```");
                fence = if in_code_block { "```" } else { "" };
                if self.return_each_line {
                    documents.push(document(line.to_owned(), metadata.clone()));
                } else {
                    current_content.push(line.to_owned());
                }
                continue;
            }

            if is_fence_line(trimmed, "~~~") {
                in_code_block = !matches!(fence, "~~~");
                fence = if in_code_block { "~~~" } else { "" };
                if self.return_each_line {
                    documents.push(document(line.to_owned(), metadata.clone()));
                } else {
                    current_content.push(line.to_owned());
                }
                continue;
            }

            if in_code_block {
                if self.return_each_line {
                    documents.push(document(line.to_owned(), metadata.clone()));
                } else {
                    current_content.push(line.to_owned());
                }
                continue;
            }

            if let Some((level, name, data)) = self.match_header(trimmed) {
                flush_markdown_section(&mut current_content, &metadata, &mut documents);

                while header_stack
                    .last()
                    .is_some_and(|header| header.level >= level)
                {
                    header_stack.pop();
                }
                header_stack.push(HeaderType { level, name, data });

                metadata = header_stack
                    .iter()
                    .map(|header| (header.name.clone(), Value::String(header.data.clone())))
                    .collect();

                if !self.strip_headers {
                    if self.return_each_line {
                        documents.push(document(line.to_owned(), metadata.clone()));
                    } else {
                        current_content.push(line.to_owned());
                    }
                }
                continue;
            }

            if self.return_each_line {
                if !trimmed.is_empty() {
                    documents.push(document(line.to_owned(), metadata.clone()));
                }
            } else if !line.is_empty() || !current_content.is_empty() {
                current_content.push(line.to_owned());
            }
        }

        flush_markdown_section(&mut current_content, &metadata, &mut documents);
        documents
    }

    fn match_header(&self, line: &str) -> Option<(usize, String, String)> {
        for (separator, name) in &self.headers_to_split_on {
            if !line.starts_with(separator) {
                continue;
            }
            if line.len() > separator.len()
                && !matches!(line.as_bytes().get(separator.len()), Some(b' '))
            {
                continue;
            }

            return Some((
                separator
                    .chars()
                    .filter(|character| *character == '#')
                    .count()
                    .max(1),
                name.clone(),
                line[separator.len()..].trim().to_owned(),
            ));
        }

        None
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExperimentalMarkdownSyntaxTextSplitter {
    inner: MarkdownHeaderTextSplitter,
}

impl ExperimentalMarkdownSyntaxTextSplitter {
    pub fn new(return_each_line: bool, strip_headers: bool) -> Self {
        Self {
            inner: MarkdownHeaderTextSplitter::new(
                default_headers(),
                return_each_line,
                strip_headers,
            ),
        }
    }

    pub fn split_text(&self, text: &str) -> Vec<Document> {
        self.inner.split_text(text)
    }
}

fn default_headers() -> Vec<(String, String)> {
    vec![
        ("#".to_owned(), "header_1".to_owned()),
        ("##".to_owned(), "header_2".to_owned()),
        ("###".to_owned(), "header_3".to_owned()),
        ("####".to_owned(), "header_4".to_owned()),
        ("#####".to_owned(), "header_5".to_owned()),
        ("######".to_owned(), "header_6".to_owned()),
    ]
}

fn is_fence_line(line: &str, fence: &str) -> bool {
    line.starts_with(fence)
}

fn flush_markdown_section(
    current_content: &mut Vec<String>,
    metadata: &BTreeMap<String, Value>,
    documents: &mut Vec<Document>,
) {
    let page_content = current_content.join("\n").trim().to_owned();
    if !page_content.is_empty() {
        documents.push(document(page_content, metadata.clone()));
    }
    current_content.clear();
}

fn document(page_content: String, metadata: BTreeMap<String, Value>) -> Document {
    Document {
        page_content,
        metadata,
        id: None,
    }
}
