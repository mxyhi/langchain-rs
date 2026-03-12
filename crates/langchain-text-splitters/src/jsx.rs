use crate::{Language, RecursiveCharacterTextSplitter, TextSplitter};

#[derive(Debug, Clone)]
pub struct JSFrameworkTextSplitter {
    inner: RecursiveCharacterTextSplitter,
}

impl JSFrameworkTextSplitter {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            inner: RecursiveCharacterTextSplitter::from_language(
                Language::Jsx,
                chunk_size,
                chunk_overlap,
            ),
        }
    }
}

impl TextSplitter for JSFrameworkTextSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        self.inner.split_text(text)
    }
}

pub type JsxTextSplitter = JSFrameworkTextSplitter;
