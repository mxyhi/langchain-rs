use reqwest::{Client, RequestBuilder};
use secrecy::{ExposeSecret, SecretString};

const ANTHROPIC_VERSION: &str = "2023-06-01";

#[derive(Debug, Clone)]
pub(crate) struct AnthropicClientConfig {
    client: Client,
    base_url: String,
    api_key: Option<SecretString>,
}

impl AnthropicClientConfig {
    pub(crate) fn new(
        client: Client,
        base_url: impl Into<String>,
        api_key: Option<impl AsRef<str>>,
    ) -> Self {
        Self {
            client,
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            api_key: api_key.map(|value| SecretString::new(value.as_ref().to_owned().into())),
        }
    }

    pub(crate) fn post(&self, path: &str) -> RequestBuilder {
        let builder = self
            .client
            .post(format!(
                "{}/{}",
                self.base_url,
                path.trim_start_matches('/')
            ))
            .header("anthropic-version", ANTHROPIC_VERSION);

        match &self.api_key {
            Some(api_key) => builder.header("x-api-key", api_key.expose_secret()),
            None => builder,
        }
    }

    pub(crate) fn base_url(&self) -> &str {
        &self.base_url
    }
}
