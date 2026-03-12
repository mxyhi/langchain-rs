use crate::TextSplitter;
use crate::utils::{group_segments, split_sentences};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KonlpyTextSplitter {
    chunk_size: usize,
    chunk_overlap: usize,
}

impl KonlpyTextSplitter {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            chunk_size: chunk_size.max(1),
            chunk_overlap,
        }
    }
}

impl TextSplitter for KonlpyTextSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        // KoNLPy usually produces sentence- or phrase-like units for Korean text.
        // A punctuation-aware fallback is cheap and still useful for mixed prose.
        let sentences = split_sentences(text, true);
        group_segments(sentences, self.chunk_size, self.chunk_overlap, " ")
    }
}
