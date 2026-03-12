use std::collections::BTreeMap;

use futures_util::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::LangChainError;
use crate::messages::{AIMessage, BaseMessage, ResponseMetadata, UsageMetadata};
use crate::output_parsers::JsonOutputKeyToolsParser;
use crate::outputs::{Generation, GenerationCandidate, LLMResult};
use crate::runnables::{Runnable, RunnableConfig, RunnableDyn};
use crate::tools::ToolDefinition;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Any,
    Named(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct StructuredOutputSchema {
    name: String,
    description: Option<String>,
    schema: Value,
}

impl StructuredOutputSchema {
    pub fn new(name: impl Into<String>, schema: Value) -> Self {
        Self {
            name: name.into(),
            description: None,
            schema,
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn schema(&self) -> &Value {
        &self.schema
    }

    pub fn to_tool_definition(&self, strict: Option<bool>) -> ToolDefinition {
        let definition = ToolDefinition::new(
            self.name.clone(),
            self.description.clone().unwrap_or_default(),
        )
        .with_parameters(self.schema.clone());

        match strict {
            Some(strict) => definition.with_strict(strict),
            None => definition,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuredOutputMethod {
    FunctionCalling,
    JsonMode,
    JsonSchema,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolBindingOptions {
    pub tool_choice: Option<ToolChoice>,
    pub strict: Option<bool>,
    pub parallel_tool_calls: Option<bool>,
    pub response_format: Option<StructuredOutputSchema>,
}

impl Default for ToolBindingOptions {
    fn default() -> Self {
        Self {
            tool_choice: None,
            strict: None,
            parallel_tool_calls: None,
            response_format: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuredOutputOptions {
    pub method: StructuredOutputMethod,
    pub include_raw: bool,
    pub strict: Option<bool>,
}

impl Default for StructuredOutputOptions {
    fn default() -> Self {
        Self {
            method: StructuredOutputMethod::FunctionCalling,
            include_raw: false,
            strict: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StructuredOutput {
    Parsed(Value),
    Raw {
        raw: AIMessage,
        parsed: Option<Value>,
        parsing_error: Option<String>,
    },
}

struct StructuredOutputRunnable {
    model: Box<dyn BaseChatModel>,
    parser: JsonOutputKeyToolsParser,
    include_raw: bool,
}

impl StructuredOutputRunnable {
    fn new(
        model: Box<dyn BaseChatModel>,
        parser: JsonOutputKeyToolsParser,
        include_raw: bool,
    ) -> Self {
        Self {
            model,
            parser,
            include_raw,
        }
    }
}

impl Runnable<Vec<BaseMessage>, StructuredOutput> for StructuredOutputRunnable {
    fn invoke<'a>(
        &'a self,
        input: Vec<BaseMessage>,
        config: RunnableConfig,
    ) -> BoxFuture<'a, Result<StructuredOutput, LangChainError>> {
        Box::pin(async move {
            let raw = self.model.generate(input, config.clone()).await?;
            match self.parser.invoke(raw.clone(), config).await {
                Ok(parsed) if self.include_raw => Ok(StructuredOutput::Raw {
                    raw,
                    parsed: Some(parsed),
                    parsing_error: None,
                }),
                Ok(parsed) => Ok(StructuredOutput::Parsed(parsed)),
                Err(error) if self.include_raw => Ok(StructuredOutput::Raw {
                    raw,
                    parsed: None,
                    parsing_error: Some(error.to_string()),
                }),
                Err(error) => Err(error),
            }
        })
    }
}

pub trait BaseChatModel: Send + Sync {
    fn model_name(&self) -> &str;

    fn generate<'a>(
        &'a self,
        messages: Vec<BaseMessage>,
        config: RunnableConfig,
    ) -> BoxFuture<'a, Result<AIMessage, LangChainError>>;

    fn identifying_params(&self) -> BTreeMap<String, serde_json::Value> {
        BTreeMap::new()
    }

    fn bind_tools(
        &self,
        _tools: Vec<ToolDefinition>,
        _options: ToolBindingOptions,
    ) -> Result<Box<dyn BaseChatModel>, LangChainError> {
        Err(LangChainError::unsupported(format!(
            "chat model `{}` does not implement bind_tools",
            self.model_name()
        )))
    }

    fn with_structured_output(
        &self,
        schema: StructuredOutputSchema,
        options: StructuredOutputOptions,
    ) -> Result<Box<dyn RunnableDyn<Vec<BaseMessage>, StructuredOutput>>, LangChainError> {
        if options.method != StructuredOutputMethod::FunctionCalling {
            return Err(LangChainError::unsupported(
                "only function_calling structured output is implemented",
            ));
        }

        let model = self.bind_tools(
            vec![schema.to_tool_definition(options.strict)],
            ToolBindingOptions {
                tool_choice: Some(ToolChoice::Any),
                strict: options.strict,
                ..ToolBindingOptions::default()
            },
        )?;

        Ok(Box::new(StructuredOutputRunnable::new(
            model,
            JsonOutputKeyToolsParser::new(schema.name()),
            options.include_raw,
        )))
    }
}

pub type BaseLanguageModel = dyn BaseChatModel;

impl<T> BaseChatModel for Box<T>
where
    T: BaseChatModel + ?Sized,
{
    fn model_name(&self) -> &str {
        (**self).model_name()
    }

    fn generate<'a>(
        &'a self,
        messages: Vec<BaseMessage>,
        config: RunnableConfig,
    ) -> BoxFuture<'a, Result<AIMessage, LangChainError>> {
        (**self).generate(messages, config)
    }

    fn identifying_params(&self) -> BTreeMap<String, serde_json::Value> {
        (**self).identifying_params()
    }

    fn bind_tools(
        &self,
        tools: Vec<ToolDefinition>,
        options: ToolBindingOptions,
    ) -> Result<Box<dyn BaseChatModel>, LangChainError> {
        (**self).bind_tools(tools, options)
    }

    fn with_structured_output(
        &self,
        schema: StructuredOutputSchema,
        options: StructuredOutputOptions,
    ) -> Result<Box<dyn RunnableDyn<Vec<BaseMessage>, StructuredOutput>>, LangChainError> {
        (**self).with_structured_output(schema, options)
    }
}

impl<T> Runnable<Vec<BaseMessage>, AIMessage> for T
where
    T: BaseChatModel,
{
    fn invoke<'a>(
        &'a self,
        input: Vec<BaseMessage>,
        config: RunnableConfig,
    ) -> BoxFuture<'a, Result<AIMessage, LangChainError>> {
        self.generate(input, config)
    }
}

pub trait BaseLLM: Send + Sync {
    fn model_name(&self) -> &str;

    fn generate<'a>(
        &'a self,
        prompts: Vec<String>,
        config: RunnableConfig,
    ) -> BoxFuture<'a, Result<LLMResult, LangChainError>>;

    fn invoke_prompt<'a>(
        &'a self,
        prompt: String,
        config: RunnableConfig,
    ) -> BoxFuture<'a, Result<String, LangChainError>> {
        Box::pin(async move {
            let response = self.generate(vec![prompt], config).await?;
            response
                .primary_generation()
                .map(GenerationCandidate::text)
                .map(ToOwned::to_owned)
                .ok_or_else(|| LangChainError::request("llm response contained no generations"))
        })
    }

    fn identifying_params(&self) -> BTreeMap<String, serde_json::Value> {
        BTreeMap::new()
    }
}

impl<T> BaseLLM for Box<T>
where
    T: BaseLLM + ?Sized,
{
    fn model_name(&self) -> &str {
        (**self).model_name()
    }

    fn generate<'a>(
        &'a self,
        prompts: Vec<String>,
        config: RunnableConfig,
    ) -> BoxFuture<'a, Result<LLMResult, LangChainError>> {
        (**self).generate(prompts, config)
    }

    fn invoke_prompt<'a>(
        &'a self,
        prompt: String,
        config: RunnableConfig,
    ) -> BoxFuture<'a, Result<String, LangChainError>> {
        (**self).invoke_prompt(prompt, config)
    }

    fn identifying_params(&self) -> BTreeMap<String, serde_json::Value> {
        (**self).identifying_params()
    }
}

impl<T> Runnable<String, String> for T
where
    T: BaseLLM,
{
    fn invoke<'a>(
        &'a self,
        input: String,
        config: RunnableConfig,
    ) -> BoxFuture<'a, Result<String, LangChainError>> {
        self.invoke_prompt(input, config)
    }
}

#[derive(Debug, Clone)]
pub struct ParrotChatModel {
    model_name: String,
    parrot_buffer_length: usize,
}

impl ParrotChatModel {
    pub fn new(model_name: impl Into<String>, parrot_buffer_length: usize) -> Self {
        Self {
            model_name: model_name.into(),
            parrot_buffer_length,
        }
    }
}

impl BaseChatModel for ParrotChatModel {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn generate<'a>(
        &'a self,
        messages: Vec<BaseMessage>,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<AIMessage, LangChainError>> {
        Box::pin(async move {
            let last_message = messages.last().ok_or(LangChainError::EmptyMessages)?;
            let content = last_message
                .content()
                .chars()
                .take(self.parrot_buffer_length)
                .collect::<String>();
            let input_tokens = messages.iter().map(|message| message.content().len()).sum();
            let output_tokens = content.len();

            let mut metadata = ResponseMetadata::new();
            metadata.insert("model".to_owned(), self.model_name.clone().into());

            Ok(AIMessage::with_metadata(
                content,
                metadata,
                Some(UsageMetadata {
                    input_tokens,
                    output_tokens,
                    total_tokens: input_tokens + output_tokens,
                }),
            ))
        })
    }

    fn identifying_params(&self) -> BTreeMap<String, serde_json::Value> {
        BTreeMap::from([(
            "model_name".to_owned(),
            serde_json::Value::String(self.model_name.clone()),
        )])
    }
}

#[derive(Debug, Clone)]
pub struct ParrotLLM {
    model_name: String,
    parrot_buffer_length: usize,
}

impl ParrotLLM {
    pub fn new(model_name: impl Into<String>, parrot_buffer_length: usize) -> Self {
        Self {
            model_name: model_name.into(),
            parrot_buffer_length,
        }
    }
}

impl BaseLLM for ParrotLLM {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn generate<'a>(
        &'a self,
        prompts: Vec<String>,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<LLMResult, LangChainError>> {
        Box::pin(async move {
            let prompt_tokens = prompts.iter().map(String::len).sum::<usize>();
            let generations = prompts
                .into_iter()
                .map(|prompt| {
                    let text = prompt
                        .chars()
                        .take(self.parrot_buffer_length)
                        .collect::<String>();
                    vec![GenerationCandidate::from(Generation::new(text))]
                })
                .collect::<Vec<_>>();
            let completion_tokens = generations
                .iter()
                .flat_map(|generation_group| generation_group.iter())
                .map(GenerationCandidate::text)
                .map(str::len)
                .sum::<usize>();

            let llm_output = BTreeMap::from([
                (
                    "model".to_owned(),
                    serde_json::Value::String(self.model_name.clone()),
                ),
                (
                    "model_name".to_owned(),
                    serde_json::Value::String(self.model_name.clone()),
                ),
                (
                    "token_usage".to_owned(),
                    serde_json::json!({
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    }),
                ),
            ]);

            Ok(LLMResult::new(generations).with_output(llm_output))
        })
    }

    fn identifying_params(&self) -> BTreeMap<String, serde_json::Value> {
        BTreeMap::from([(
            "model_name".to_owned(),
            serde_json::Value::String(self.model_name.clone()),
        )])
    }
}
