use reqwest::Client;
use serde::{Deserialize, Serialize};

use langchain_core::LangChainError;

use crate::client::OpenAIClientConfig;

#[derive(Debug, Clone)]
pub struct OpenAIEmbeddings {
    config: OpenAIClientConfig,
    model: String,
    dimensions: Option<usize>,
    chunk_size: usize,
}

impl OpenAIEmbeddings {
    pub fn new(
        model: impl Into<String>,
        base_url: impl Into<String>,
        api_key: Option<impl AsRef<str>>,
    ) -> Self {
        Self {
            config: OpenAIClientConfig::new(Client::new(), base_url, api_key),
            model: model.into(),
            dimensions: None,
            chunk_size: 128,
        }
    }

    pub fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = Some(dimensions);
        self
    }

    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size.max(1);
        self
    }

    pub fn base_url(&self) -> &str {
        self.config.base_url()
    }

    pub async fn embed_query(&self, text: impl AsRef<str>) -> Result<Vec<f32>, LangChainError> {
        let mut embeddings = self.embed_documents([text.as_ref()]).await?;
        embeddings.pop().ok_or_else(|| {
            LangChainError::request("openai embeddings response contained no vectors")
        })
    }

    pub async fn embed_documents<I, S>(&self, texts: I) -> Result<Vec<Vec<f32>>, LangChainError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let texts = texts
            .into_iter()
            .map(|value| value.as_ref().to_owned())
            .collect::<Vec<_>>();
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(self.chunk_size) {
            let request = EmbeddingsRequest {
                model: self.model.clone(),
                input: chunk.to_vec(),
                dimensions: self.dimensions,
            };

            let response = self
                .config
                .post("embeddings")
                .json(&request)
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

            let response = response
                .json::<EmbeddingsResponse>()
                .await
                .map_err(|error| LangChainError::request(error.to_string()))?;

            let mut vectors = response
                .data
                .into_iter()
                .map(|item| item.embedding)
                .collect::<Vec<_>>();
            all_embeddings.append(&mut vectors);
        }

        Ok(all_embeddings)
    }
}

#[derive(Debug, Clone, Serialize)]
struct EmbeddingsRequest {
    model: String,
    input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Clone, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}
