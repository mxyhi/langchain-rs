use langchain_core::LangChainError;

const BUILTIN_PROVIDERS: &[&str] = &["openai"];

pub fn init_embeddings(
    model: &str,
    provider: Option<&str>,
    base_url: impl Into<String>,
    api_key: Option<&str>,
) -> Result<Box<dyn Embeddings>, LangChainError> {
    let (provider, model) = infer_model_and_provider(model, provider)?;
    match provider.as_str() {
        "openai" => Ok(Box::new(langchain_openai::OpenAIEmbeddings::new(
            model, base_url, api_key,
        ))),
        _ => Err(LangChainError::unsupported(format!(
            "Provider '{provider}' is not supported.\nSupported providers and their required packages:\n{}",
            provider_list()
        ))),
    }
}

fn parse_model_string(model_name: &str) -> Result<(String, String), LangChainError> {
    if !model_name.contains(':') {
        return Err(LangChainError::request(format!(
            "Invalid model format '{model_name}'.\nModel name must be in format 'provider:model-name'\nSupported providers: {:?}",
            BUILTIN_PROVIDERS
        )));
    }

    let (provider, model) = model_name
        .split_once(':')
        .expect("model format should contain ':'");
    let provider = provider.trim().to_ascii_lowercase();
    let model = model.trim().to_owned();

    if !BUILTIN_PROVIDERS.contains(&provider.as_str()) {
        return Err(LangChainError::unsupported(format!(
            "Provider '{provider}' is not supported.\nSupported providers and their required packages:\n{}",
            provider_list()
        )));
    }

    if model.is_empty() {
        return Err(LangChainError::request("Model name cannot be empty"));
    }

    Ok((provider, model))
}

fn infer_model_and_provider(
    model: &str,
    provider: Option<&str>,
) -> Result<(String, String), LangChainError> {
    let model = model.trim();
    if model.is_empty() {
        return Err(LangChainError::request("Model name cannot be empty"));
    }

    if let Some(provider) = provider.map(|value| value.trim().to_ascii_lowercase()) {
        if provider.is_empty() {
            return Err(LangChainError::request(format!(
                "Must specify either:\n1. A model string in format 'provider:model-name'\n2. Or explicitly set provider from: {:?}",
                BUILTIN_PROVIDERS
            )));
        }
        if !BUILTIN_PROVIDERS.contains(&provider.as_str()) {
            return Err(LangChainError::unsupported(format!(
                "Provider '{provider}' is not supported.\nSupported providers and their required packages:\n{}",
                provider_list()
            )));
        }
        return Ok((provider, model.to_owned()));
    }

    if model.contains(':') {
        return parse_model_string(model);
    }

    Err(LangChainError::request(format!(
        "Must specify either:\n1. A model string in format 'provider:model-name'\n2. Or explicitly set provider from: {:?}",
        BUILTIN_PROVIDERS
    )))
}

fn provider_list() -> String {
    BUILTIN_PROVIDERS
        .iter()
        .map(|provider| format!("  - {provider}: langchain-openai"))
        .collect::<Vec<_>>()
        .join("\n")
}

pub use langchain_core::embeddings::{CharacterEmbeddings, Embeddings};
pub use langchain_openai::OpenAIEmbeddings;
