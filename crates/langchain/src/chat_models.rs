use langchain_core::LangChainError;

const BUILTIN_PROVIDERS: &[&str] = &["openai"];

pub fn init_chat_model(
    model: &str,
    model_provider: Option<&str>,
    base_url: impl Into<String>,
    api_key: Option<&str>,
) -> Result<Box<dyn BaseChatModel>, LangChainError> {
    let (model, provider) = parse_model(model, model_provider)?;
    match provider.as_str() {
        "openai" => Ok(Box::new(langchain_openai::ChatOpenAI::new(
            model, base_url, api_key,
        ))),
        _ => Err(LangChainError::unsupported(format!(
            "Unsupported provider='{provider}'. Supported providers are: {}",
            supported_providers()
        ))),
    }
}

pub fn attempt_infer_model_provider(model_name: &str) -> Option<&'static str> {
    let model_name = model_name.trim().to_ascii_lowercase();

    if model_name.starts_with("gpt-")
        || model_name.starts_with("o1")
        || model_name.starts_with("o3")
        || model_name.starts_with("chatgpt")
        || model_name.starts_with("text-davinci")
    {
        Some("openai")
    } else {
        None
    }
}

fn parse_model(
    model: &str,
    model_provider: Option<&str>,
) -> Result<(String, String), LangChainError> {
    let mut model = model.trim().to_owned();
    let mut model_provider = model_provider.map(normalize_provider);

    // Keep Python LangChain's rule: only strip the first provider prefix when it is
    // recognized, leaving the rest of the model string untouched.
    if model_provider.is_none()
        && let Some((provider, parsed_model)) = split_provider_model(&model)
    {
        model_provider = Some(provider);
        model = parsed_model;
    }

    let model_provider = model_provider
        .or_else(|| attempt_infer_model_provider(&model).map(str::to_owned))
        .ok_or_else(|| {
            LangChainError::unsupported(format!(
                "Unable to infer model provider for model={model:?}. Please specify 'model_provider' directly.\n\nSupported providers: {}",
                supported_providers()
            ))
        })?;

    if !BUILTIN_PROVIDERS.contains(&model_provider.as_str()) {
        return Err(LangChainError::unsupported(format!(
            "Unsupported provider='{model_provider}'. Supported providers are: {}",
            supported_providers()
        )));
    }

    if model.is_empty() {
        return Err(LangChainError::request("Model name cannot be empty"));
    }

    Ok((model, model_provider))
}

fn split_provider_model(model: &str) -> Option<(String, String)> {
    let (provider, model_name) = model.split_once(':')?;
    let provider = normalize_provider(provider);
    BUILTIN_PROVIDERS
        .contains(&provider.as_str())
        .then_some((provider, model_name.to_owned()))
}

fn normalize_provider(provider: &str) -> String {
    provider.replace('-', "_").trim().to_ascii_lowercase()
}

fn supported_providers() -> String {
    BUILTIN_PROVIDERS.join(", ")
}

pub use langchain_core::language_models::{BaseChatModel, ParrotChatModel};
pub use langchain_openai::ChatOpenAI;
