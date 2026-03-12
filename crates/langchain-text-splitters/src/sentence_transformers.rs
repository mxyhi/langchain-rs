use crate::{TextSplitter, TokenTextSplitter, Tokenizer};

#[derive(Debug, Clone)]
pub struct SentenceTransformersTokenTextSplitter {
    inner: TokenTextSplitter,
}

impl SentenceTransformersTokenTextSplitter {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            inner: TokenTextSplitter::new(Tokenizer::whitespace(), chunk_size, chunk_overlap),
        }
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        self.inner.tokenizer()
    }
}

impl TextSplitter for SentenceTransformersTokenTextSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        self.inner.split_text(text)
    }
}
