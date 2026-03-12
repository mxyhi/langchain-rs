use crate::base::TextSplitter;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CharacterTextSplitter {
    separator: String,
    chunk_size: usize,
    chunk_overlap: usize,
}

impl CharacterTextSplitter {
    pub fn new(separator: String, chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            separator,
            chunk_size,
            chunk_overlap,
        }
    }
}

impl TextSplitter for CharacterTextSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        let pieces = if self.separator.is_empty() {
            text.chars().map(|value| value.to_string()).collect()
        } else {
            text.split(&self.separator)
                .filter(|part| !part.is_empty())
                .map(str::to_owned)
                .collect()
        };

        merge_splits(pieces, self.chunk_size, &self.separator)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecursiveCharacterTextSplitter {
    chunk_size: usize,
    chunk_overlap: usize,
    separators: Vec<String>,
}

impl RecursiveCharacterTextSplitter {
    pub fn new(chunk_size: usize, chunk_overlap: usize, separators: Vec<String>) -> Self {
        Self {
            chunk_size,
            chunk_overlap,
            separators,
        }
    }

    fn split_with_separator(&self, text: &str, separator_index: usize) -> Vec<String> {
        if text.chars().count() <= self.chunk_size {
            return vec![text.to_owned()];
        }

        let Some(separator) = self.separators.get(separator_index) else {
            return fallback_split(text, self.chunk_size, self.chunk_overlap);
        };

        if separator.is_empty() || !text.contains(separator) {
            return self.split_with_separator(text, separator_index + 1);
        }

        let pieces = text
            .split(separator)
            .filter(|part| !part.is_empty())
            .flat_map(|part| {
                if part.chars().count() > self.chunk_size {
                    self.split_with_separator(part, separator_index + 1)
                } else {
                    vec![part.to_owned()]
                }
            })
            .collect::<Vec<_>>();

        let chunks = merge_splits(pieces, self.chunk_size, separator);
        if chunks
            .iter()
            .all(|chunk| chunk.chars().count() <= self.chunk_size)
        {
            chunks
        } else {
            fallback_split(text, self.chunk_size, self.chunk_overlap)
        }
    }
}

impl TextSplitter for RecursiveCharacterTextSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        self.split_with_separator(text, 0)
    }
}

fn merge_splits(pieces: Vec<String>, chunk_size: usize, separator: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();

    for piece in pieces {
        let candidate = if current.is_empty() {
            piece.clone()
        } else {
            format!("{current}{separator}{piece}")
        };

        if candidate.chars().count() <= chunk_size {
            current = candidate;
            continue;
        }

        if !current.is_empty() {
            chunks.push(current);
            current = String::new();
        }

        current.push_str(&piece);
    }

    if !current.is_empty() {
        chunks.push(current);
    }

    chunks
        .into_iter()
        .map(|chunk| chunk.trim().to_owned())
        .filter(|chunk| !chunk.is_empty())
        .collect()
}

fn fallback_split(text: &str, chunk_size: usize, chunk_overlap: usize) -> Vec<String> {
    let characters = text.chars().collect::<Vec<_>>();
    let mut chunks = Vec::new();
    let step = chunk_size.saturating_sub(chunk_overlap).max(1);
    let mut index = 0;

    while index < characters.len() {
        let end = (index + chunk_size).min(characters.len());
        chunks.push(characters[index..end].iter().collect::<String>());
        if end == characters.len() {
            break;
        }
        index += step;
    }

    chunks
}
