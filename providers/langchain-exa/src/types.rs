use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextContentsOptions {
    pub max_characters: Option<usize>,
}

impl TextContentsOptions {
    pub fn with_max_characters(mut self, max_characters: usize) -> Self {
        self.max_characters = Some(max_characters);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HighlightsContentsOptions {
    pub highlights_per_url: usize,
}

impl Default for HighlightsContentsOptions {
    fn default() -> Self {
        Self {
            highlights_per_url: 3,
        }
    }
}

impl HighlightsContentsOptions {
    pub fn with_highlights_per_url(mut self, highlights_per_url: usize) -> Self {
        self.highlights_per_url = highlights_per_url.max(1);
        self
    }
}
