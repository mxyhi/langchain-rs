use langchain_text_splitters::{TextSplitter, TokenTextSplitter, Tokenizer, split_text_on_tokens};

#[test]
fn split_text_on_tokens_respects_chunk_size_and_overlap() {
    let tokenizer = Tokenizer::whitespace();
    let chunks = split_text_on_tokens("alpha beta gamma delta", &tokenizer, 2, 1);

    assert_eq!(chunks, vec!["alpha beta", "beta gamma", "gamma delta"]);
}

#[test]
fn token_text_splitter_creates_documents_from_token_windows() {
    let splitter = TokenTextSplitter::new(Tokenizer::whitespace(), 2, 0);
    let documents = splitter.create_documents(["alpha beta gamma"], None);

    assert_eq!(documents.len(), 2);
    assert_eq!(documents[0].page_content, "alpha beta");
    assert_eq!(documents[1].page_content, "gamma");
}
