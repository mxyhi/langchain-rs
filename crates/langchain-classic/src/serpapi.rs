use langchain_core::LangChainError;
use serde_json::Value;

fn map_reqwest_error(error: reqwest::Error) -> LangChainError {
    LangChainError::request(error.to_string())
}

#[derive(Debug, Clone)]
pub struct SerpAPIWrapper {
    api_key: String,
    endpoint: String,
    engine: String,
    client: reqwest::Client,
}

impl SerpAPIWrapper {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            endpoint: "https://serpapi.com/search".to_owned(),
            engine: "google".to_owned(),
            client: reqwest::Client::new(),
        }
    }

    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    pub fn with_engine(mut self, engine: impl Into<String>) -> Self {
        self.engine = engine.into();
        self
    }

    pub async fn search(&self, query: &str) -> Result<Value, LangChainError> {
        self.client
            .get(&self.endpoint)
            .query(&[
                ("q", query),
                ("api_key", self.api_key.as_str()),
                ("engine", self.engine.as_str()),
            ])
            .send()
            .await
            .map_err(map_reqwest_error)?
            .error_for_status()
            .map_err(map_reqwest_error)?
            .json()
            .await
            .map_err(map_reqwest_error)
    }
}
