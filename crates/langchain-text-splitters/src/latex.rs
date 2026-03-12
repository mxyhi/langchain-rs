use crate::{Language, RecursiveCharacterTextSplitter, TextSplitter};

#[derive(Debug, Clone)]
pub struct LatexTextSplitter {
    inner: RecursiveCharacterTextSplitter,
}

impl LatexTextSplitter {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            inner: RecursiveCharacterTextSplitter::from_language(
                Language::Latex,
                chunk_size,
                chunk_overlap,
            ),
        }
    }
}

impl TextSplitter for LatexTextSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        self.inner.split_text(text)
    }
}
