use langchain_core::LangChainError;
use langchain_core::runnables::RunnableConfig;
use serde_json::{Value, json};

pub const CACHE_CONTROL_CONFIG_KEY: &str = "anthropic.cache_control";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnthropicPromptCachingMiddleware {
    ttl: String,
    min_messages_to_cache: usize,
}

impl Default for AnthropicPromptCachingMiddleware {
    fn default() -> Self {
        Self {
            ttl: "5m".to_owned(),
            min_messages_to_cache: 0,
        }
    }
}

impl AnthropicPromptCachingMiddleware {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_ttl(mut self, ttl: impl Into<String>) -> Self {
        self.ttl = ttl.into();
        self
    }

    pub fn with_min_messages_to_cache(mut self, min_messages_to_cache: usize) -> Self {
        self.min_messages_to_cache = min_messages_to_cache;
        self
    }

    pub fn ttl(&self) -> &str {
        &self.ttl
    }

    pub fn min_messages_to_cache(&self) -> usize {
        self.min_messages_to_cache
    }

    pub fn should_apply(&self, message_count: usize) -> bool {
        message_count >= self.min_messages_to_cache
    }

    pub fn cache_control(&self) -> Result<Value, LangChainError> {
        if !matches!(self.ttl.as_str(), "5m" | "1h") {
            return Err(LangChainError::request(format!(
                "unsupported Anthropic prompt cache TTL `{}`; expected `5m` or `1h`",
                self.ttl
            )));
        }

        Ok(json!({
            "type": "ephemeral",
            "ttl": self.ttl
        }))
    }

    pub fn configured_config(
        &self,
        message_count: usize,
        mut config: RunnableConfig,
    ) -> Result<RunnableConfig, LangChainError> {
        if self.should_apply(message_count) {
            config
                .metadata
                .insert(CACHE_CONTROL_CONFIG_KEY.to_owned(), self.cache_control()?);
        }

        Ok(config)
    }
}
