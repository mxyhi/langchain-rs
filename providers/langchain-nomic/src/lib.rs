use futures_util::future::BoxFuture;
use langchain_core::LangChainError;
use langchain_core::embeddings::Embeddings;

#[derive(Debug, Clone)]
pub struct NomicEmbeddings {
    model: String,
}

impl NomicEmbeddings {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
        }
    }

    pub fn model(&self) -> &str {
        &self.model
    }
}

impl Embeddings for NomicEmbeddings {
    fn embed_query<'a>(
        &'a self,
        _text: &'a str,
    ) -> BoxFuture<'a, Result<Vec<f32>, LangChainError>> {
        Box::pin(async move {
            Err(LangChainError::unsupported(
                "NomicEmbeddings transport is not implemented in this milestone",
            ))
        })
    }

    fn embed_documents<'a>(
        &'a self,
        _texts: Vec<String>,
    ) -> BoxFuture<'a, Result<Vec<Vec<f32>>, LangChainError>> {
        Box::pin(async move {
            Err(LangChainError::unsupported(
                "NomicEmbeddings transport is not implemented in this milestone",
            ))
        })
    }
}
