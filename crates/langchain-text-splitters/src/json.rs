use crate::{Language, RecursiveCharacterTextSplitter, TextSplitter};

#[derive(Debug, Clone)]
pub struct RecursiveJsonSplitter {
    inner: RecursiveCharacterTextSplitter,
}

impl RecursiveJsonSplitter {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            inner: RecursiveCharacterTextSplitter::from_language(
                Language::Json,
                chunk_size,
                chunk_overlap,
            ),
        }
    }
}

impl TextSplitter for RecursiveJsonSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        self.inner.split_text(text)
    }
}

pub type JsonTextSplitter = RecursiveJsonSplitter;
