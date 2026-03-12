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

const BUILTIN_PROVIDERS: &[&str] = &["openai"];

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
    base_url: String,
    api_key: Option<String>,
    queued_declarative_operations: Vec<DeclarativeOperation>,
}

impl ConfigurableChatModel {
    pub fn new(
        default_model: Option<impl Into<String>>,
        default_provider: Option<impl Into<String>>,
        base_url: impl Into<String>,
        api_key: Option<impl Into<String>>,
    ) -> Self {
        Self {
            default_model: default_model.map(Into::into),
            default_provider: default_provider.map(Into::into),
            base_url: base_url.into(),
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
    ) -> Result<(String, Option<String>), LangChainError> {
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
            .map(normalize_provider)
            .or_else(|| self.default_provider.clone());

        Ok((model, provider))
    }

    fn resolve_bound_model(
        &self,
        config: &RunnableConfig,
    ) -> Result<Box<dyn BaseChatModel>, LangChainError> {
        let (model, provider) = self.resolve_model_from_config(config)?;
        let mut resolved = init_chat_model(
            &model,
            provider.as_deref(),
            self.base_url.clone(),
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

pub fn init_configurable_chat_model(
    default_model: Option<&str>,
    model_provider: Option<&str>,
    base_url: impl Into<String>,
    api_key: Option<&str>,
) -> ConfigurableChatModel {
    ConfigurableChatModel::new(default_model, model_provider, base_url, api_key)
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

pub use langchain_core::language_models::{ParrotChatModel, StructuredOutputMethod, ToolChoice};
pub use langchain_openai::ChatOpenAI;
