use futures_util::future::BoxFuture;
use langchain_core::LangChainError;
use langchain_core::embeddings::Embeddings;
use reqwest::Client;
use serde::{Deserialize, Serialize};

const DEFAULT_BASE_URL: &str = "https://api-atlas.nomic.ai";

#[derive(Debug, Clone)]
pub struct NomicEmbeddings {
    client: Client,
    model: String,
    base_url: String,
    api_key: Option<String>,
    dimensionality: Option<usize>,
}

impl NomicEmbeddings {
    pub fn new(model: impl Into<String>) -> Self {
        Self::new_with_base_url(model, DEFAULT_BASE_URL, std::env::var("NOMIC_API_KEY").ok())
    }

    pub fn new_with_base_url(
        model: impl Into<String>,
        base_url: impl Into<String>,
        api_key: Option<impl AsRef<str>>,
    ) -> Self {
        Self {
            client: Client::new(),
            model: model.into(),
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            api_key: api_key.map(|value| value.as_ref().to_owned()),
            dimensionality: None,
        }
    }

    pub fn with_dimensionality(mut self, dimensionality: usize) -> Self {
        self.dimensionality = Some(dimensionality);
        self
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    async fn embed_texts(
        &self,
        texts: Vec<String>,
        task_type: &'static str,
    ) -> Result<Vec<Vec<f32>>, LangChainError> {
        let mut request = self
            .client
            .post(format!("{}/v1/embedding/text", self.base_url))
            .json(&EmbeddingRequest {
                model: self.model.clone(),
                texts,
                task_type,
                dimensionality: self.dimensionality,
            });

        if let Some(api_key) = &self.api_key {
            request = request.bearer_auth(api_key);
        }

        let response = request
            .send()
            .await
            .map_err(|error| LangChainError::request(error.to_string()))?;
        let status = response.status();
        if !status.is_success() {
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| String::from("<unreadable body>"));
            return Err(LangChainError::HttpStatus {
                status: status.as_u16(),
                body,
            });
        }

        response
            .json::<EmbeddingResponse>()
            .await
            .map(|response| response.embeddings)
            .map_err(|error| LangChainError::request(error.to_string()))
    }
}

impl Embeddings for NomicEmbeddings {
    fn embed_query<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Vec<f32>, LangChainError>> {
        Box::pin(async move {
            let mut embeddings = self
                .embed_texts(vec![text.to_owned()], "search_query")
                .await?;
            embeddings.pop().ok_or_else(|| {
                LangChainError::request("nomic embeddings response contained no vectors")
            })
        })
    }

    fn embed_documents<'a>(
        &'a self,
        texts: Vec<String>,
    ) -> BoxFuture<'a, Result<Vec<Vec<f32>>, LangChainError>> {
        Box::pin(async move { self.embed_texts(texts, "search_document").await })
    }
}

#[derive(Debug, Clone, Serialize)]
struct EmbeddingRequest {
    model: String,
    texts: Vec<String>,
    task_type: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensionality: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct EmbeddingResponse {
    embeddings: Vec<Vec<f32>>,
}

pub mod embeddings {
    pub use crate::NomicEmbeddings;
}
