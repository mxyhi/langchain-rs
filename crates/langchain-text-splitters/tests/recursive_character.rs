use langchain_text_splitters::{
    CharacterTextSplitter, RecursiveCharacterTextSplitter, TextSplitter,
};

#[test]
fn recursive_splitter_honors_chunk_size_and_overlap() {
    let splitter = RecursiveCharacterTextSplitter::new(6, 2, vec![" ".to_owned(), "".to_owned()]);
    let chunks = splitter.split_text("alpha beta gamma");

    assert!(chunks.len() >= 3);
    assert_eq!(chunks.first().expect("first chunk"), "alpha");
    assert_eq!(chunks.last().expect("last chunk"), "gamma");
    assert!(chunks.iter().all(|chunk| chunk.chars().count() <= 6));
}

#[test]
fn character_splitter_can_create_documents() {
    let splitter = CharacterTextSplitter::new(" ".to_owned(), 5, 1);
    let documents = splitter.create_documents(["one two three"], None);

    assert_eq!(documents.len(), 3);
    assert_eq!(documents[0].page_content, "one");
    assert_eq!(documents[1].page_content, "two");
    assert_eq!(documents[2].page_content, "three");
}
