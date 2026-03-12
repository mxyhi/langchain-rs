use futures_util::future::BoxFuture;
use langchain_core::LangChainError;

pub trait EmbeddingsUnderTest: Send + Sync {
    fn embed_query<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Vec<f32>, LangChainError>>;

    fn embed_documents<'a>(
        &'a self,
        texts: Vec<String>,
    ) -> BoxFuture<'a, Result<Vec<Vec<f32>>, LangChainError>>;
}

pub async fn assert_embedding_dimension<E>(embeddings: &E, text: &str, expected_dimension: usize)
where
    E: EmbeddingsUnderTest,
{
    let embedding = embeddings
        .embed_query(text)
        .await
        .expect("query embedding should succeed");

    assert_eq!(embedding.len(), expected_dimension);
}

pub async fn assert_document_embeddings_count<E>(
    embeddings: &E,
    texts: Vec<String>,
) -> Vec<Vec<f32>>
where
    E: EmbeddingsUnderTest,
{
    let results = embeddings
        .embed_documents(texts.clone())
        .await
        .expect("document embeddings should succeed");

    assert_eq!(results.len(), texts.len());
    results
}

pub async fn assert_query_and_documents_share_dimension<E>(
    embeddings: &E,
    query: &str,
    documents: Vec<String>,
) where
    E: EmbeddingsUnderTest,
{
    let query_embedding = embeddings
        .embed_query(query)
        .await
        .expect("query embedding should succeed");
    let document_embeddings = embeddings
        .embed_documents(documents)
        .await
        .expect("document embeddings should succeed");

    for embedding in document_embeddings {
        assert_eq!(embedding.len(), query_embedding.len());
    }
}
