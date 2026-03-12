use langchain_core::documents::Document;
use std::fmt;
use std::sync::Arc;

pub trait TextSplitter {
    fn split_text(&self, text: &str) -> Vec<String>;

    fn create_documents<I, S>(
        &self,
        texts: I,
        _metadatas: Option<Vec<serde_json::Value>>,
    ) -> Vec<Document>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        texts
            .into_iter()
            .flat_map(|text| self.split_text(text.as_ref()))
            .map(Document::new)
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    PlainText,
    Markdown,
    Html,
    Json,
    Jsx,
    Latex,
    Python,
    JavaScript,
}

type TokenEncoder = dyn Fn(&str) -> Vec<String> + Send + Sync;
type TokenDecoder = dyn Fn(&[String]) -> String + Send + Sync;

#[derive(Clone)]
pub struct Tokenizer {
    encoder: Arc<TokenEncoder>,
    decoder: Arc<TokenDecoder>,
}

impl fmt::Debug for Tokenizer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Tokenizer").finish_non_exhaustive()
    }
}

impl Tokenizer {
    pub fn new(
        encoder: impl Fn(&str) -> Vec<String> + Send + Sync + 'static,
        decoder: impl Fn(&[String]) -> String + Send + Sync + 'static,
    ) -> Self {
        Self {
            encoder: Arc::new(encoder),
            decoder: Arc::new(decoder),
        }
    }

    pub fn whitespace() -> Self {
        Self::new(
            |text| text.split_whitespace().map(str::to_owned).collect(),
            |tokens| tokens.join(" "),
        )
    }

    pub fn character() -> Self {
        Self::new(
            |text| text.chars().map(|value| value.to_string()).collect(),
            |tokens| tokens.concat(),
        )
    }

    pub fn encode(&self, text: &str) -> Vec<String> {
        (self.encoder)(text)
    }

    pub fn decode(&self, tokens: &[String]) -> String {
        (self.decoder)(tokens)
    }
}

#[derive(Debug, Clone)]
pub struct TokenTextSplitter {
    tokenizer: Tokenizer,
    chunk_size: usize,
    chunk_overlap: usize,
}

impl TokenTextSplitter {
    pub fn new(tokenizer: Tokenizer, chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            tokenizer,
            chunk_size: chunk_size.max(1),
            chunk_overlap,
        }
    }

    pub fn from_language(language: Language, chunk_size: usize, chunk_overlap: usize) -> Self {
        let tokenizer = match language {
            Language::PlainText
            | Language::Markdown
            | Language::Html
            | Language::Json
            | Language::Jsx
            | Language::Latex
            | Language::Python
            | Language::JavaScript => Tokenizer::whitespace(),
        };

        Self::new(tokenizer, chunk_size, chunk_overlap)
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
}

impl TextSplitter for TokenTextSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        split_text_on_tokens(text, &self.tokenizer, self.chunk_size, self.chunk_overlap)
    }
}

pub fn split_text_on_tokens(
    text: &str,
    tokenizer: &Tokenizer,
    chunk_size: usize,
    chunk_overlap: usize,
) -> Vec<String> {
    let chunk_size = chunk_size.max(1);
    let tokens = tokenizer.encode(text);
    if tokens.is_empty() {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let step = chunk_size.saturating_sub(chunk_overlap).max(1);
    let mut index = 0;

    while index < tokens.len() {
        let end = (index + chunk_size).min(tokens.len());
        let chunk = tokenizer.decode(&tokens[index..end]);
        if !chunk.is_empty() {
            chunks.push(chunk);
        }
        if end == tokens.len() {
            break;
        }
        index += step;
    }

    chunks
}
