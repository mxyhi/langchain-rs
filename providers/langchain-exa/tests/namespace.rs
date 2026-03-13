use langchain_core::runnables::Runnable;

#[tokio::test]
async fn exa_namespaces_match_root_exports() {
    let root_tool = langchain_exa::ExaSearchResults::new()
        .with_hit(langchain_exa::SearchHit::new(
            "Rust async guide",
            "https://example.com/rust",
            "Rust async runtime and futures",
            0.0,
        ))
        .with_text_options(langchain_exa::TextContentsOptions::default().with_max_characters(10));
    let namespaced_tool = langchain_exa::tools::ExaSearchResults::new()
        .with_hit(langchain_exa::tools::SearchHit::new(
            "Rust async guide",
            "https://example.com/rust",
            "Rust async runtime and futures",
            0.0,
        ))
        .with_text_options(
            langchain_exa::types::TextContentsOptions::default().with_max_characters(10),
        );
    assert_eq!(
        root_tool
            .invoke("rust".to_owned(), Default::default())
            .await
            .expect("root tool should succeed"),
        namespaced_tool
            .invoke("rust".to_owned(), Default::default())
            .await
            .expect("namespaced tool should succeed")
    );

    let root_retriever =
        langchain_exa::ExaSearchRetriever::new().with_hit(langchain_exa::SearchHit::new(
            "LangChain",
            "https://example.com/langchain",
            "Build retrieval pipelines in Rust",
            0.0,
        ));
    let namespaced_retriever = langchain_exa::retrievers::ExaSearchRetriever::new().with_hit(
        langchain_exa::tools::SearchHit::new(
            "LangChain",
            "https://example.com/langchain",
            "Build retrieval pipelines in Rust",
            0.0,
        ),
    );
    assert_eq!(
        root_retriever
            .invoke("retrieval".to_owned(), Default::default())
            .await
            .expect("root retriever should succeed"),
        namespaced_retriever
            .invoke("retrieval".to_owned(), Default::default())
            .await
            .expect("namespaced retriever should succeed")
    );

    let root_finder =
        langchain_exa::ExaFindSimilarResults::new().with_hit(langchain_exa::SearchHit::new(
            "Agent search",
            "https://example.com/agent",
            "Agentic search workflows",
            0.0,
        ));
    let namespaced_finder = langchain_exa::tools::ExaFindSimilarResults::new().with_hit(
        langchain_exa::tools::SearchHit::new(
            "Agent search",
            "https://example.com/agent",
            "Agentic search workflows",
            0.0,
        ),
    );
    assert_eq!(
        root_finder
            .invoke("agent".to_owned(), Default::default())
            .await
            .expect("root similarity tool should succeed"),
        namespaced_finder
            .invoke("agent".to_owned(), Default::default())
            .await
            .expect("namespaced similarity tool should succeed")
    );
}
