use std::collections::BTreeMap;

use futures_util::future::BoxFuture;
use langchain_core::LangChainError;
use langchain_core::language_models::{
    BaseChatModel, StructuredOutput, StructuredOutputOptions, StructuredOutputSchema,
    ToolBindingOptions,
};
use langchain_core::messages::{AIMessage, BaseMessage};
use langchain_core::runnables::{Runnable, RunnableConfig, RunnableDyn};
use langchain_core::tools::ToolDefinition;
use langchain_model_profiles::{
    infer_chat_provider, normalize_provider_key, provider, supported_chat_providers,
};

#[derive(Debug, Clone)]
enum DeclarativeOperation {
    BindTools {
        tools: Vec<ToolDefinition>,
        options: ToolBindingOptions,
    },
}

#[derive(Debug, Clone)]
pub struct ConfigurableChatModel {
    default_model: Option<String>,
    default_provider: Option<String>,
    base_url: Option<String>,
    api_key: Option<String>,
    queued_declarative_operations: Vec<DeclarativeOperation>,
}

impl ConfigurableChatModel {
    pub fn new(
        default_model: Option<impl Into<String>>,
        default_provider: Option<impl Into<String>>,
        base_url: Option<impl Into<String>>,
        api_key: Option<impl Into<String>>,
    ) -> Self {
        Self {
            default_model: default_model.map(Into::into),
            default_provider: default_provider.map(Into::into),
            base_url: base_url.map(Into::into),
            api_key: api_key.map(Into::into),
            queued_declarative_operations: Vec::new(),
        }
    }

    pub fn queued_operation_count(&self) -> usize {
        self.queued_declarative_operations.len()
    }

    pub fn bind_tools(mut self, tools: Vec<ToolDefinition>, options: ToolBindingOptions) -> Self {
        self.queued_declarative_operations
            .push(DeclarativeOperation::BindTools { tools, options });
        self
    }

    pub fn with_structured_output(
        &self,
        schema: StructuredOutputSchema,
        options: StructuredOutputOptions,
    ) -> ConfigurableStructuredOutputRunnable {
        ConfigurableStructuredOutputRunnable {
            model: self.clone(),
            schema,
            options,
        }
    }

    fn resolve_model_from_config(
        &self,
        config: &RunnableConfig,
    ) -> Result<(String, Option<String>, Option<String>), LangChainError> {
        let model = config
            .configurable
            .get("model")
            .and_then(|value| value.as_str())
            .map(str::to_owned)
            .or_else(|| self.default_model.clone())
            .ok_or_else(|| {
                LangChainError::unsupported(
                    "configurable chat model requires `configurable.model` or a default model",
                )
            })?;

        let provider = config
            .configurable
            .get("provider")
            .and_then(|value| value.as_str())
            .map(normalize_provider_key)
            .or_else(|| self.default_provider.clone());
        let runtime_base_url = config
            .configurable
            .get("base_url")
            .and_then(|value| value.as_str())
            .map(str::to_owned);

        Ok((model, provider, runtime_base_url))
    }

    fn resolve_bound_model(
        &self,
        config: &RunnableConfig,
    ) -> Result<Box<dyn BaseChatModel>, LangChainError> {
        let (model, provider, runtime_base_url) = self.resolve_model_from_config(config)?;
        let base_url = self.base_url.as_deref().or(runtime_base_url.as_deref());
        let mut resolved = init_chat_model(
            &model,
            provider.as_deref(),
            base_url,
            self.api_key.as_deref(),
        )?;

        for operation in &self.queued_declarative_operations {
            match operation {
                DeclarativeOperation::BindTools { tools, options } => {
                    resolved = resolved.bind_tools(tools.clone(), options.clone())?;
                }
            }
        }

        Ok(resolved)
    }
}

#[derive(Debug, Clone)]
pub struct ConfigurableStructuredOutputRunnable {
    model: ConfigurableChatModel,
    schema: StructuredOutputSchema,
    options: StructuredOutputOptions,
}

impl Runnable<Vec<BaseMessage>, StructuredOutput> for ConfigurableStructuredOutputRunnable {
    fn invoke<'a>(
        &'a self,
        input: Vec<BaseMessage>,
        config: RunnableConfig,
    ) -> BoxFuture<'a, Result<StructuredOutput, LangChainError>> {
        Box::pin(async move {
            let resolved = self.model.resolve_bound_model(&config)?;
            let runnable =
                resolved.with_structured_output(self.schema.clone(), self.options.clone())?;
            runnable.invoke_boxed(input, config).await
        })
    }
}

impl BaseChatModel for ConfigurableChatModel {
    fn model_name(&self) -> &str {
        self.default_model
            .as_deref()
            .unwrap_or("configurable-chat-model")
    }

    fn generate<'a>(
        &'a self,
        messages: Vec<BaseMessage>,
        config: RunnableConfig,
    ) -> BoxFuture<'a, Result<AIMessage, LangChainError>> {
        Box::pin(async move {
            let resolved = self.resolve_bound_model(&config)?;
            resolved.generate(messages, config).await
        })
    }

    fn bind_tools(
        &self,
        tools: Vec<ToolDefinition>,
        options: ToolBindingOptions,
    ) -> Result<Box<dyn BaseChatModel>, LangChainError> {
        Ok(Box::new(ConfigurableChatModel::bind_tools(
            self.clone(),
            tools,
            options,
        )))
    }

    fn with_structured_output(
        &self,
        schema: StructuredOutputSchema,
        options: StructuredOutputOptions,
    ) -> Result<Box<dyn RunnableDyn<Vec<BaseMessage>, StructuredOutput>>, LangChainError> {
        Ok(Box::new(ConfigurableChatModel::with_structured_output(
            self, schema, options,
        )))
    }

    fn identifying_params(&self) -> BTreeMap<String, serde_json::Value> {
        let mut params = BTreeMap::new();
        if let Some(model) = &self.default_model {
            params.insert("model".to_owned(), serde_json::Value::String(model.clone()));
        }
        if let Some(provider) = &self.default_provider {
            params.insert(
                "provider".to_owned(),
                serde_json::Value::String(provider.clone()),
            );
        }
        if let Some(base_url) = &self.base_url {
            params.insert(
                "base_url".to_owned(),
                serde_json::Value::String(base_url.clone()),
            );
        }
        params.insert(
            "queued_operations".to_owned(),
            serde_json::Value::from(self.queued_declarative_operations.len() as u64),
        );
        params
    }
}

pub fn init_chat_model(
    model: &str,
    model_provider: Option<&str>,
    base_url: Option<&str>,
    api_key: Option<&str>,
) -> Result<Box<dyn BaseChatModel>, LangChainError> {
    let (model, provider_key) = parse_model(model, model_provider)?;
    let resolved_base_url = resolve_base_url(&provider_key, base_url);

    match provider_key.as_str() {
        "anthropic" => Ok(Box::new(langchain_anthropic::ChatAnthropic::new(
            model,
            require_base_url(&provider_key, resolved_base_url.as_deref())?,
            api_key,
        ))),
        "deepseek" => Ok(Box::new(
            langchain_deepseek::ChatDeepSeek::new_with_base_url(
                model,
                require_base_url(&provider_key, resolved_base_url.as_deref())?,
                api_key,
            ),
        )),
        "fireworks" => Ok(Box::new(
            langchain_fireworks::ChatFireworks::new_with_base_url(
                model,
                require_base_url(&provider_key, resolved_base_url.as_deref())?,
                api_key,
            ),
        )),
        "groq" => Ok(Box::new(langchain_groq::ChatGroq::new_with_base_url(
            model,
            require_base_url(&provider_key, resolved_base_url.as_deref())?,
            api_key,
        ))),
        "huggingface" => Ok(Box::new(
            langchain_huggingface::ChatHuggingFace::from_model_id(model),
        )),
        "mistralai" => Ok(Box::new(
            langchain_mistralai::ChatMistralAI::new_with_base_url(
                model,
                require_base_url(&provider_key, resolved_base_url.as_deref())?,
                api_key,
            ),
        )),
        "ollama" => Ok(Box::new(langchain_ollama::ChatOllama::new_with_base_url(
            model,
            require_base_url(&provider_key, resolved_base_url.as_deref())?,
            api_key,
        ))),
        "openai" => Ok(Box::new(langchain_openai::ChatOpenAI::new(
            model,
            require_base_url(&provider_key, resolved_base_url.as_deref())?,
            api_key,
        ))),
        "openrouter" => Ok(Box::new(
            langchain_openrouter::ChatOpenRouter::new_with_base_url(
                model,
                require_base_url(&provider_key, resolved_base_url.as_deref())?,
                api_key,
            ),
        )),
        "perplexity" => Ok(Box::new(langchain_perplexity::ChatPerplexity::new(model))),
        "xai" => Ok(Box::new(langchain_xai::ChatXAI::new_with_base_url(
            model,
            require_base_url(&provider_key, resolved_base_url.as_deref())?,
            api_key,
        ))),
        _ => Err(LangChainError::unsupported(format!(
            "Unsupported provider='{provider_key}'. Supported providers are: {}",
            supported_providers()
        ))),
    }
}

pub fn init_configurable_chat_model(
    default_model: Option<&str>,
    model_provider: Option<&str>,
    base_url: Option<&str>,
    api_key: Option<&str>,
) -> ConfigurableChatModel {
    ConfigurableChatModel::new(default_model, model_provider, base_url, api_key)
}

pub fn attempt_infer_model_provider(model_name: &str) -> Option<&'static str> {
    infer_chat_provider(model_name).map(|profile| profile.key)
}

fn parse_model(
    model: &str,
    model_provider: Option<&str>,
) -> Result<(String, String), LangChainError> {
    let mut model = model.trim().to_owned();
    let mut model_provider = model_provider.map(normalize_provider_key);

    if model_provider.is_none()
        && let Some((provider_key, parsed_model)) = split_provider_model(&model)
    {
        model_provider = Some(provider_key);
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

    if provider(&model_provider).is_none_or(|profile| !profile.capabilities.chat_model) {
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
    let (provider_key, model_name) = model.split_once(':')?;
    let provider_key = normalize_provider_key(provider_key);
    provider(&provider_key)
        .filter(|profile| profile.capabilities.chat_model)
        .map(|_| (provider_key, model_name.to_owned()))
}

fn resolve_base_url(provider_key: &str, explicit_base_url: Option<&str>) -> Option<String> {
    explicit_base_url.map(str::to_owned).or_else(|| {
        provider(provider_key).and_then(|profile| profile.default_base_url.map(str::to_owned))
    })
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

fn supported_providers() -> String {
    supported_chat_providers().join(", ")
}

pub use langchain_anthropic::ChatAnthropic;
pub use langchain_core::language_models::{ParrotChatModel, StructuredOutputMethod, ToolChoice};
pub use langchain_deepseek::ChatDeepSeek;
pub use langchain_fireworks::ChatFireworks;
pub use langchain_groq::ChatGroq;
pub use langchain_huggingface::ChatHuggingFace;
pub use langchain_mistralai::ChatMistralAI;
pub use langchain_ollama::ChatOllama;
pub use langchain_openai::ChatOpenAI;
pub use langchain_openrouter::ChatOpenRouter;
pub use langchain_perplexity::ChatPerplexity;
pub use langchain_xai::ChatXAI;
