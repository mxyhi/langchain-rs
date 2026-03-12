use langchain_core::LangChainError;
use langchain_model_profiles::{normalize_provider_key, provider, supported_embedding_providers};

pub fn init_embeddings(
    model: &str,
    provider_key: Option<&str>,
    base_url: Option<&str>,
    api_key: Option<&str>,
) -> Result<Box<dyn Embeddings>, LangChainError> {
    let (provider_key, model) = infer_model_and_provider(model, provider_key)?;
    let resolved_base_url = resolve_base_url(&provider_key, base_url);

    match provider_key.as_str() {
        "fireworks" => Ok(Box::new(
            langchain_fireworks::FireworksEmbeddings::new_with_base_url(
                model,
                require_base_url(&provider_key, resolved_base_url.as_deref())?,
                api_key,
            ),
        )),
        "huggingface" => Ok(Box::new(langchain_huggingface::HuggingFaceEmbeddings::new(
            model,
        ))),
        "mistralai" => Ok(Box::new(
            langchain_mistralai::MistralAIEmbeddings::new_with_base_url(
                model,
                require_base_url(&provider_key, resolved_base_url.as_deref())?,
                api_key,
            ),
        )),
        "nomic" => Ok(Box::new(langchain_nomic::NomicEmbeddings::new(model))),
        "ollama" => Ok(Box::new(langchain_ollama::OllamaEmbeddings::new_with_base_url(
            model,
            require_base_url(&provider_key, resolved_base_url.as_deref())?,
            api_key,
        ))),
        "openai" => Ok(Box::new(langchain_openai::OpenAIEmbeddings::new(
            model,
            require_base_url(&provider_key, resolved_base_url.as_deref())?,
            api_key,
        ))),
        _ => Err(LangChainError::unsupported(format!(
            "Provider '{provider_key}' is not supported.\nSupported providers and their required packages:\n{}",
            provider_list()
        ))),
    }
}

fn parse_model_string(model_name: &str) -> Result<(String, String), LangChainError> {
    if !model_name.contains(':') {
        return Err(LangChainError::request(format!(
            "Invalid model format '{model_name}'.\nModel name must be in format 'provider:model-name'\nSupported providers: {}",
            supported_embedding_providers().join(", ")
        )));
    }

    let (provider_key, model) = model_name
        .split_once(':')
        .expect("model format should contain ':'");
    let provider_key = normalize_provider_key(provider_key);
    let model = model.trim().to_owned();

    if provider(&provider_key)
        .is_none_or(|profile| !profile.capabilities.embeddings)
    {
        return Err(LangChainError::unsupported(format!(
            "Provider '{provider_key}' is not supported.\nSupported providers and their required packages:\n{}",
            provider_list()
        )));
    }

    if model.is_empty() {
        return Err(LangChainError::request("Model name cannot be empty"));
    }

    Ok((provider_key, model))
}

fn infer_model_and_provider(
    model: &str,
    provider_key: Option<&str>,
) -> Result<(String, String), LangChainError> {
    let model = model.trim();
    if model.is_empty() {
        return Err(LangChainError::request("Model name cannot be empty"));
    }

    if let Some(provider_key) = provider_key.map(normalize_provider_key) {
        if provider(&provider_key)
            .is_none_or(|profile| !profile.capabilities.embeddings)
        {
            return Err(LangChainError::unsupported(format!(
                "Provider '{provider_key}' is not supported.\nSupported providers and their required packages:\n{}",
                provider_list()
            )));
        }
        return Ok((provider_key, model.to_owned()));
    }

    if model.contains(':') {
        return parse_model_string(model);
    }

    Err(LangChainError::request(format!(
        "Must specify either:\n1. A model string in format 'provider:model-name'\n2. Or explicitly set provider from: {}",
        supported_embedding_providers().join(", ")
    )))
}

fn provider_list() -> String {
    supported_embedding_providers()
        .into_iter()
        .map(|provider_key| {
            let package_name = provider(provider_key)
                .map(|profile| profile.package_name)
                .unwrap_or("unknown package");
            format!("  - {provider_key}: {package_name}")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn resolve_base_url(provider_key: &str, explicit_base_url: Option<&str>) -> Option<String> {
    explicit_base_url
        .map(str::to_owned)
        .or_else(|| provider(provider_key).and_then(|profile| profile.default_base_url.map(str::to_owned)))
}

fn require_base_url(provider_key: &str, base_url: Option<&str>) -> Result<String, LangChainError> {
    base_url.map(str::to_owned).ok_or_else(|| {
        let package_name = provider(provider_key)
            .map(|profile| profile.package_name)
            .unwrap_or("unknown package");
        LangChainError::unsupported(format!(
            "provider '{provider_key}' requires a base_url or a registered default in {package_name}",
        ))
    })
}

pub use langchain_core::embeddings::{CharacterEmbeddings, Embeddings};
pub use langchain_fireworks::FireworksEmbeddings;
pub use langchain_huggingface::{HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings};
pub use langchain_mistralai::MistralAIEmbeddings;
pub use langchain_nomic::NomicEmbeddings;
pub use langchain_ollama::OllamaEmbeddings;
pub use langchain_openai::OpenAIEmbeddings;
