# langchain-text-splitters

Standalone text splitter crate for chunking plain text, markup, code, and structured documents.

## Upstream Mapping

This crate maps to `libs/text-splitters` in the reference monorepo.

## Installation

```bash
cargo add langchain-text-splitters
```

## Quick Start

```rust
use langchain_text_splitters::{RecursiveCharacterTextSplitter, TextSplitter};

let splitter = RecursiveCharacterTextSplitter::new(
    6,
    2,
    vec![" ".to_owned(), "".to_owned()],
);
let chunks = splitter.split_text("alpha beta gamma");
assert_eq!(chunks[0], "alpha");
```

## Public Surface

- Character and token splitters
- Markdown, HTML, JSON, JSX, Python, LaTeX, NLTK, spaCy, Konlpy, and sentence-transformer variants
- Header-aware and syntax-aware splitters for richer document metadata

## Tests

- `tests/recursive_character.rs`
- `tests/token_splitter.rs`
- `tests/language_modules.rs`
