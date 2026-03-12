use crate::TextSplitter;
use crate::utils::{group_segments, split_sentences};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NLTKTextSplitter {
    chunk_size: usize,
    chunk_overlap: usize,
}

impl NLTKTextSplitter {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            chunk_size: chunk_size.max(1),
            chunk_overlap,
        }
    }
}

impl TextSplitter for NLTKTextSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        let sentences = split_sentences(text, false);
        group_segments(sentences, self.chunk_size, self.chunk_overlap, " ")
    }
}
