use reqwest::{Client, RequestBuilder};
use secrecy::{ExposeSecret, SecretString};

#[derive(Debug, Clone)]
pub(crate) struct OpenAIClientConfig {
    client: Client,
    base_url: String,
    api_key: Option<SecretString>,
}

impl OpenAIClientConfig {
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
        let builder = self.client.post(format!("{}/{}", self.base_url, path));
        match &self.api_key {
            Some(api_key) => builder.bearer_auth(api_key.expose_secret()),
            None => builder,
        }
    }

    pub(crate) fn base_url(&self) -> &str {
        &self.base_url
    }
}
