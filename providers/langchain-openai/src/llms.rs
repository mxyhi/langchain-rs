use std::collections::BTreeMap;

use futures_util::future::BoxFuture;
use langchain_core::LangChainError;
use langchain_core::language_models::BaseLLM;
use langchain_core::outputs::{Generation, LLMResult};
use langchain_core::runnables::RunnableConfig;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::client::OpenAIClientConfig;

#[derive(Debug, Clone)]
pub struct OpenAI {
    config: OpenAIClientConfig,
    model: String,
    temperature: Option<f32>,
    max_tokens: Option<usize>,
    top_p: Option<f32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    n: usize,
    best_of: Option<usize>,
    seed: Option<u64>,
    logprobs: Option<u8>,
    batch_size: usize,
}

impl OpenAI {
    pub fn new(
        model: impl Into<String>,
        base_url: impl Into<String>,
        api_key: Option<impl AsRef<str>>,
    ) -> Self {
        Self {
            config: OpenAIClientConfig::new(Client::new(), base_url, api_key),
            model: model.into(),
            temperature: None,
            max_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            n: 1,
            best_of: None,
            seed: None,
            logprobs: None,
            batch_size: 20,
        }
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    pub fn with_frequency_penalty(mut self, frequency_penalty: f32) -> Self {
        self.frequency_penalty = Some(frequency_penalty);
        self
    }

    pub fn with_presence_penalty(mut self, presence_penalty: f32) -> Self {
        self.presence_penalty = Some(presence_penalty);
        self
    }

    pub fn with_n(mut self, n: usize) -> Self {
        self.n = n.max(1);
        self
    }

    pub fn with_best_of(mut self, best_of: usize) -> Self {
        self.best_of = Some(best_of.max(1));
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn with_logprobs(mut self, logprobs: u8) -> Self {
        self.logprobs = Some(logprobs);
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.max(1);
        self
    }

    pub fn base_url(&self) -> &str {
        self.config.base_url()
    }

    fn request_body(&self, prompts: Vec<String>) -> CompletionRequest {
        CompletionRequest {
            model: self.model.clone(),
            prompt: prompts,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            top_p: self.top_p,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            n: (self.n > 1).then_some(self.n),
            best_of: self.best_of,
            seed: self.seed,
            logprobs: self.logprobs,
        }
    }

    fn create_llm_result(
        &self,
        choices: Vec<CompletionChoice>,
        prompts: &[String],
        token_usage: CompletionUsage,
        system_fingerprint: Option<String>,
        response_model: Option<String>,
    ) -> Result<LLMResult, LangChainError> {
        let expected_choices = prompts.len() * self.n;
        if choices.len() < expected_choices {
            return Err(LangChainError::request(format!(
                "openai completions response contained {} choices for {} prompts with n={}",
                choices.len(),
                prompts.len(),
                self.n
            )));
        }

        // OpenAI flattens choices across prompts, so we reconstruct prompt-local groups
        // using the configured `n` in the same order as Python LangChain.
        let generations = prompts
            .iter()
            .enumerate()
            .map(|(index, _)| {
                choices[index * self.n..(index + 1) * self.n]
                    .iter()
                    .map(|choice| {
                        let mut info = BTreeMap::new();
                        info.insert(
                            "finish_reason".to_owned(),
                            choice
                                .finish_reason
                                .clone()
                                .map(Value::String)
                                .unwrap_or(Value::Null),
                        );
                        info.insert(
                            "logprobs".to_owned(),
                            choice.logprobs.clone().unwrap_or(Value::Null),
                        );

                        Generation::with_info(choice.text.clone(), info)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let mut llm_output = BTreeMap::from([
            (
                "model_name".to_owned(),
                Value::String(response_model.unwrap_or_else(|| self.model.clone())),
            ),
            (
                "token_usage".to_owned(),
                json!({
                    "prompt_tokens": token_usage.prompt_tokens,
                    "completion_tokens": token_usage.completion_tokens,
                    "total_tokens": token_usage.total_tokens,
                }),
            ),
        ]);
        if let Some(system_fingerprint) = system_fingerprint {
            llm_output.insert(
                "system_fingerprint".to_owned(),
                Value::String(system_fingerprint),
            );
        }

        Ok(LLMResult::new(generations).with_output(llm_output))
    }
}

impl BaseLLM for OpenAI {
    fn model_name(&self) -> &str {
        &self.model
    }

    fn generate<'a>(
        &'a self,
        prompts: Vec<String>,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<LLMResult, LangChainError>> {
        Box::pin(async move {
            if prompts.is_empty() {
                return Ok(LLMResult::new(Vec::<Vec<Generation>>::new()).with_output(
                    BTreeMap::from([
                        ("model_name".to_owned(), Value::String(self.model.clone())),
                        (
                            "token_usage".to_owned(),
                            json!({
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "total_tokens": 0,
                            }),
                        ),
                    ]),
                ));
            }

            let mut choices = Vec::new();
            let mut usage = CompletionUsage::default();
            let mut system_fingerprint = None;
            let mut response_model = None;

            for prompt_batch in prompts.chunks(self.batch_size) {
                let response = self
                    .config
                    .post("completions")
                    .json(&self.request_body(prompt_batch.to_vec()))
                    .send()
                    .await
                    .map_err(|error| LangChainError::request(error.to_string()))?;
                let status = response.status();

                if !status.is_success() {
                    let body = response
                        .text()
                        .await
                        .unwrap_or_else(|_| String::from("<unreadable body>"));
                    return Err(LangChainError::HttpStatus {
                        status: status.as_u16(),
                        body,
                    });
                }

                let response = response
                    .json::<CompletionResponse>()
                    .await
                    .map_err(|error| LangChainError::request(error.to_string()))?;

                if let Some(error) = response.error {
                    return Err(LangChainError::request(error.to_string()));
                }

                usage.prompt_tokens += response.usage.prompt_tokens;
                usage.completion_tokens += response.usage.completion_tokens;
                usage.total_tokens += response.usage.total_tokens;
                choices.extend(response.choices);
                response_model = response.model.or(response_model);
                if system_fingerprint.is_none() {
                    system_fingerprint = response.system_fingerprint;
                }
            }

            self.create_llm_result(choices, &prompts, usage, system_fingerprint, response_model)
        })
    }

    fn identifying_params(&self) -> BTreeMap<String, Value> {
        let mut params = BTreeMap::from([("model_name".to_owned(), json!(self.model))]);
        if let Some(temperature) = self.temperature {
            params.insert("temperature".to_owned(), json!(temperature));
        }
        if let Some(max_tokens) = self.max_tokens {
            params.insert("max_tokens".to_owned(), json!(max_tokens));
        }
        if let Some(top_p) = self.top_p {
            params.insert("top_p".to_owned(), json!(top_p));
        }
        if let Some(frequency_penalty) = self.frequency_penalty {
            params.insert("frequency_penalty".to_owned(), json!(frequency_penalty));
        }
        if let Some(presence_penalty) = self.presence_penalty {
            params.insert("presence_penalty".to_owned(), json!(presence_penalty));
        }
        if let Some(best_of) = self.best_of {
            params.insert("best_of".to_owned(), json!(best_of));
        }
        if let Some(seed) = self.seed {
            params.insert("seed".to_owned(), json!(seed));
        }
        if let Some(logprobs) = self.logprobs {
            params.insert("logprobs".to_owned(), json!(logprobs));
        }
        params.insert("n".to_owned(), json!(self.n));
        params.insert("batch_size".to_owned(), json!(self.batch_size));
        params
    }
}

#[derive(Debug, Clone, Serialize)]
struct CompletionRequest {
    model: String,
    prompt: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    n: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    best_of: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<u8>,
}

#[derive(Debug, Clone, Deserialize)]
struct CompletionResponse {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    choices: Vec<CompletionChoice>,
    #[serde(default)]
    usage: CompletionUsage,
    #[serde(default)]
    system_fingerprint: Option<String>,
    #[serde(default)]
    error: Option<Value>,
}

#[derive(Debug, Clone, Deserialize)]
struct CompletionChoice {
    text: String,
    #[serde(default)]
    finish_reason: Option<String>,
    #[serde(default)]
    logprobs: Option<Value>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct CompletionUsage {
    #[serde(default)]
    prompt_tokens: usize,
    #[serde(default)]
    completion_tokens: usize,
    #[serde(default)]
    total_tokens: usize,
}
