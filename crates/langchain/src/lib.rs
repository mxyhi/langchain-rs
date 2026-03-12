pub mod chat_models;

pub mod documents {
    pub use langchain_core::documents::*;
}

pub mod embeddings;

pub mod language_models {
    pub use langchain_core::language_models::*;
}

pub mod messages {
    pub use langchain_core::messages::*;
}

pub mod output_parsers {
    pub use langchain_core::output_parsers::*;
}

pub mod prompts {
    pub use langchain_core::prompts::*;
}

pub mod runnables {
    pub use langchain_core::runnables::*;
}

pub mod text_splitters {
    pub use langchain_text_splitters::*;
}

pub mod tools {
    pub use langchain_core::tools::*;
}

pub mod vectorstores {
    pub use langchain_core::vectorstores::*;
}

pub use langchain_core::LangChainError;

#[derive(Debug, Clone, Default)]
pub struct ModelInitOptions {
    provider: Option<String>,
    base_url: Option<String>,
    api_key: Option<String>,
}

impl ModelInitOptions {
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = Some(provider.into());
        self
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }
}

pub fn init_chat_model(
    model: &str,
    options: ModelInitOptions,
) -> Result<Box<dyn language_models::BaseChatModel>, LangChainError> {
    chat_models::init_chat_model(
        model,
        options.provider.as_deref(),
        options.base_url.as_deref(),
        options.api_key.as_deref(),
    )
}

pub fn init_configurable_chat_model(
    default_model: Option<&str>,
    options: ModelInitOptions,
) -> chat_models::ConfigurableChatModel {
    chat_models::init_configurable_chat_model(
        default_model,
        options.provider.as_deref(),
        options.base_url.as_deref(),
        options.api_key.as_deref(),
    )
}

pub fn init_embeddings(
    model: &str,
    options: ModelInitOptions,
) -> Result<Box<dyn embeddings::Embeddings>, LangChainError> {
    if options.provider.is_none() && !model.contains(':') {
        return Err(LangChainError::unsupported(
            "must specify provider or use `provider:model` format for embeddings",
        ));
    }

    embeddings::init_embeddings(
        model,
        options.provider.as_deref(),
        options.base_url.as_deref(),
        options.api_key.as_deref(),
    )
}
