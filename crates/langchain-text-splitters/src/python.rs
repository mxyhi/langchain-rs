use crate::{Language, RecursiveCharacterTextSplitter, TextSplitter};

#[derive(Debug, Clone)]
pub struct PythonCodeTextSplitter {
    inner: RecursiveCharacterTextSplitter,
}

impl PythonCodeTextSplitter {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            inner: RecursiveCharacterTextSplitter::from_language(
                Language::Python,
                chunk_size,
                chunk_overlap,
            ),
        }
    }
}

impl TextSplitter for PythonCodeTextSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        self.inner.split_text(text)
    }
}
