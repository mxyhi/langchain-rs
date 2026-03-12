use crate::TextSplitter;
use crate::utils::{group_segments, split_sentences};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpacyTextSplitter {
    chunk_size: usize,
    chunk_overlap: usize,
}

impl SpacyTextSplitter {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            chunk_size: chunk_size.max(1),
            chunk_overlap,
        }
    }
}

impl TextSplitter for SpacyTextSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        // spaCy is sentence-boundary aware; using newline-sensitive segmentation
        // keeps list- and paragraph-heavy prose more stable than plain punctuation.
        let sentences = split_sentences(text, true);
        group_segments(sentences, self.chunk_size, self.chunk_overlap, " ")
    }
}
