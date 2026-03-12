use langchain_core::LangChainError;
use reqwest::Response;
use serde_json::Value;

fn map_reqwest_error(error: reqwest::Error) -> LangChainError {
    LangChainError::request(error.to_string())
}

#[derive(Debug, Clone, Default)]
pub struct Requests {
    client: reqwest::Client,
}

impl Requests {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn get(&self, url: &str) -> Result<Response, LangChainError> {
        self.client
            .get(url)
            .send()
            .await
            .map_err(map_reqwest_error)?
            .error_for_status()
            .map_err(map_reqwest_error)
    }

    pub async fn post_json(&self, url: &str, body: &Value) -> Result<Response, LangChainError> {
        self.client
            .post(url)
            .json(body)
            .send()
            .await
            .map_err(map_reqwest_error)?
            .error_for_status()
            .map_err(map_reqwest_error)
    }
}

#[derive(Debug, Clone, Default)]
pub struct RequestsWrapper {
    requests: Requests,
}

impl RequestsWrapper {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_requests(mut self, requests: Requests) -> Self {
        self.requests = requests;
        self
    }

    pub async fn get_text(&self, url: &str) -> Result<String, LangChainError> {
        self.requests
            .get(url)
            .await?
            .text()
            .await
            .map_err(map_reqwest_error)
    }

    pub async fn get_json(&self, url: &str) -> Result<Value, LangChainError> {
        self.requests
            .get(url)
            .await?
            .json()
            .await
            .map_err(map_reqwest_error)
    }

    pub async fn post_json(&self, url: &str, body: &Value) -> Result<String, LangChainError> {
        self.requests
            .post_json(url, body)
            .await?
            .text()
            .await
            .map_err(map_reqwest_error)
    }
}

pub type TextRequestsWrapper = RequestsWrapper;
