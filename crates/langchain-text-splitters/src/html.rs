use crate::{Language, RecursiveCharacterTextSplitter, TextSplitter};
use langchain_core::documents::Document;
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElementType {
    pub tag: String,
    pub content: String,
    pub metadata: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HTMLHeaderTextSplitter {
    headers_to_split_on: Vec<(String, String)>,
    return_each_element: bool,
}

impl HTMLHeaderTextSplitter {
    pub fn new(headers_to_split_on: Vec<(String, String)>, return_each_element: bool) -> Self {
        let mut headers_to_split_on = headers_to_split_on;
        headers_to_split_on.sort_by_key(|(tag, _)| header_level(tag));
        Self {
            headers_to_split_on,
            return_each_element,
        }
    }

    pub fn split_text(&self, text: &str) -> Vec<Document> {
        let spans = extract_header_spans(text, &self.headers_to_split_on);
        if spans.is_empty() {
            return vec![document(text.trim().to_owned(), BTreeMap::new())];
        }

        let mut active_headers = Vec::new();
        let mut documents = Vec::new();

        for (index, span) in spans.iter().enumerate() {
            active_headers.retain(|(level, _, _): &(usize, String, String)| *level < span.level);
            active_headers.push((
                span.level,
                span.metadata_key.clone(),
                span.header_text.clone(),
            ));

            let metadata = active_headers
                .iter()
                .map(|(_, key, value)| (key.clone(), Value::String(value.clone())))
                .collect::<BTreeMap<_, _>>();
            let section_end = spans.get(index + 1).map_or(text.len(), |next| next.start);
            let raw_section = text[span.start..section_end].trim().to_owned();

            if self.return_each_element {
                let elements = extract_elements(&raw_section, &metadata);
                if elements.is_empty() {
                    documents.push(document(raw_section, metadata));
                } else {
                    documents.extend(
                        elements
                            .into_iter()
                            .map(|element| document(element.content, element.metadata)),
                    );
                }
            } else {
                documents.push(document(raw_section, metadata));
            }
        }

        documents
    }
}

#[derive(Debug, Clone)]
pub struct HTMLSectionSplitter {
    inner: RecursiveCharacterTextSplitter,
}

impl HTMLSectionSplitter {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            inner: RecursiveCharacterTextSplitter::from_language(
                Language::Html,
                chunk_size,
                chunk_overlap,
            ),
        }
    }
}

impl TextSplitter for HTMLSectionSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        self.inner.split_text(text)
    }
}

#[derive(Debug, Clone)]
pub struct HTMLSemanticPreservingSplitter {
    inner: RecursiveCharacterTextSplitter,
}

impl HTMLSemanticPreservingSplitter {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            inner: RecursiveCharacterTextSplitter::from_language(
                Language::Html,
                chunk_size,
                chunk_overlap,
            ),
        }
    }
}

impl TextSplitter for HTMLSemanticPreservingSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        self.inner.split_text(text)
    }
}

#[derive(Debug, Clone)]
struct HeaderSpan {
    start: usize,
    level: usize,
    metadata_key: String,
    header_text: String,
}

fn extract_header_spans(text: &str, headers_to_split_on: &[(String, String)]) -> Vec<HeaderSpan> {
    let lower_text = text.to_ascii_lowercase();
    let mut spans = Vec::new();
    let mut index = 0;

    while index < lower_text.len() {
        let Some(relative_start) = lower_text[index..].find('<') else {
            break;
        };
        let start = index + relative_start;
        let mut matched = false;

        for (tag, metadata_key) in headers_to_split_on {
            if !matches_header_tag(&lower_text, start, tag) {
                continue;
            }

            let Some(open_end) = lower_text[start..].find('>').map(|offset| start + offset) else {
                continue;
            };
            let closing_tag = format!("</{tag}>");
            let content_start = open_end + 1;
            let Some(close_start) = lower_text[content_start..]
                .find(&closing_tag)
                .map(|offset| content_start + offset)
            else {
                continue;
            };

            spans.push(HeaderSpan {
                start,
                level: header_level(tag),
                metadata_key: metadata_key.clone(),
                header_text: strip_tags(&text[content_start..close_start]),
            });

            index = close_start + closing_tag.len();
            matched = true;
            break;
        }

        if !matched {
            index = start + 1;
        }
    }

    spans
}

fn matches_header_tag(lower_text: &str, start: usize, tag: &str) -> bool {
    let candidate = format!("<{tag}");
    if !lower_text[start..].starts_with(&candidate) {
        return false;
    }

    matches!(
        lower_text.as_bytes().get(start + candidate.len()).copied(),
        Some(b'>') | Some(b' ') | Some(b'\t') | Some(b'\n')
    )
}

fn extract_elements(section: &str, metadata: &BTreeMap<String, Value>) -> Vec<ElementType> {
    let mut elements = Vec::new();

    // A full DOM parser would be overkill here; this lightweight pass is enough
    // to keep paragraph-like blocks separate when callers request per-element docs.
    for closing_tag in ["</p>", "</li>", "</div>", "</section>", "</article>"] {
        if !section.contains(closing_tag) {
            continue;
        }

        let mut start = 0;
        while let Some(offset) = section[start..].find(closing_tag) {
            let end = start + offset + closing_tag.len();
            let candidate = section[start..end].trim();
            if !candidate.is_empty() {
                elements.push(ElementType {
                    tag: closing_tag
                        .trim_start_matches("</")
                        .trim_end_matches('>')
                        .to_owned(),
                    content: candidate.to_owned(),
                    metadata: metadata.clone(),
                });
            }
            start = end;
        }

        if !elements.is_empty() {
            return elements;
        }
    }

    elements
}

fn strip_tags(text: &str) -> String {
    let mut content = String::new();
    let mut in_tag = false;

    for character in text.chars() {
        match character {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => content.push(character),
            _ => {}
        }
    }

    content.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn header_level(tag: &str) -> usize {
    tag.trim_start_matches('h')
        .parse::<usize>()
        .unwrap_or(usize::MAX)
}

fn document(page_content: String, metadata: BTreeMap<String, Value>) -> Document {
    Document {
        page_content,
        metadata,
        id: None,
    }
}
