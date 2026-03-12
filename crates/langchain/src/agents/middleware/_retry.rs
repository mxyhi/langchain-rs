use std::future::Future;
use std::time::Duration;

use futures_util::future::BoxFuture;
use langchain_core::LangChainError;
use langchain_core::messages::ToolMessage;

use super::types::{
    AgentMiddleware, ModelCallHandler, ModelRequest, ModelResponse, ToolCallHandler,
    ToolCallRequest,
};

#[derive(Debug, Clone, PartialEq, Eq)]
struct RetryPolicy {
    max_retries: usize,
    base_delay: Duration,
    max_delay: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 2,
            base_delay: Duration::from_millis(50),
            max_delay: Duration::from_secs(2),
        }
    }
}

impl RetryPolicy {
    fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }

    fn with_base_delay(mut self, base_delay: Duration) -> Self {
        self.base_delay = base_delay;
        self
    }

    fn with_max_delay(mut self, max_delay: Duration) -> Self {
        self.max_delay = max_delay.max(self.base_delay);
        self
    }

    fn delay_for_attempt(&self, attempt: usize) -> Duration {
        if attempt == 0 {
            return Duration::ZERO;
        }

        let factor = 1_u32
            .checked_shl((attempt.saturating_sub(1)).min(16) as u32)
            .unwrap_or(u32::MAX);
        self.base_delay.saturating_mul(factor).min(self.max_delay)
    }

    fn should_retry(&self, error: &LangChainError) -> bool {
        match error {
            LangChainError::Request { .. } => true,
            LangChainError::HttpStatus { status, .. } => {
                matches!(status, 408 | 409 | 425 | 429 | 500 | 502 | 503 | 504)
            }
            _ => false,
        }
    }

    async fn run<T, Op, Fut>(&self, mut operation: Op) -> Result<T, LangChainError>
    where
        Op: FnMut(usize) -> Fut,
        Fut: Future<Output = Result<T, LangChainError>>,
    {
        let mut attempt = 0;

        loop {
            attempt += 1;
            match operation(attempt).await {
                Ok(value) => return Ok(value),
                Err(error)
                    if attempt <= self.max_retries.saturating_add(1)
                        && attempt <= self.max_retries
                        && self.should_retry(&error) =>
                {
                    // Middleware lives in the sync crate layer; blocking sleep keeps it runtime-agnostic.
                    std::thread::sleep(self.delay_for_attempt(attempt));
                }
                Err(error) => return Err(error),
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ModelRetryMiddleware {
    policy: RetryPolicy,
}

impl ModelRetryMiddleware {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.policy = self.policy.clone().with_max_retries(max_retries);
        self
    }

    pub fn with_base_delay(mut self, base_delay: Duration) -> Self {
        self.policy = self.policy.clone().with_base_delay(base_delay);
        self
    }

    pub fn with_max_delay(mut self, max_delay: Duration) -> Self {
        self.policy = self.policy.clone().with_max_delay(max_delay);
        self
    }

    pub fn max_retries(&self) -> usize {
        self.policy.max_retries
    }

    pub fn delay_for_attempt(&self, attempt: usize) -> Duration {
        self.policy.delay_for_attempt(attempt)
    }

    pub fn should_retry(&self, error: &LangChainError) -> bool {
        self.policy.should_retry(error)
    }

    pub async fn retry_model_call<T, Op, Fut>(&self, operation: Op) -> Result<T, LangChainError>
    where
        Op: FnMut(usize) -> Fut,
        Fut: Future<Output = Result<T, LangChainError>>,
    {
        self.policy.run(operation).await
    }
}

impl AgentMiddleware for ModelRetryMiddleware {
    fn wrap_model_call(
        &self,
        request: ModelRequest,
        handler: ModelCallHandler,
    ) -> BoxFuture<'static, Result<ModelResponse, LangChainError>> {
        let retry = self.clone();
        Box::pin(async move {
            retry
                .retry_model_call(|_| {
                    let request = request.clone();
                    let handler = handler.clone();
                    async move { handler(request).await }
                })
                .await
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ToolRetryMiddleware {
    policy: RetryPolicy,
}

impl ToolRetryMiddleware {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.policy = self.policy.clone().with_max_retries(max_retries);
        self
    }

    pub fn with_base_delay(mut self, base_delay: Duration) -> Self {
        self.policy = self.policy.clone().with_base_delay(base_delay);
        self
    }

    pub fn with_max_delay(mut self, max_delay: Duration) -> Self {
        self.policy = self.policy.clone().with_max_delay(max_delay);
        self
    }

    pub fn max_retries(&self) -> usize {
        self.policy.max_retries
    }

    pub fn should_retry(&self, error: &LangChainError) -> bool {
        self.policy.should_retry(error)
    }

    pub async fn retry_tool_call<T, Op, Fut>(&self, operation: Op) -> Result<T, LangChainError>
    where
        Op: FnMut(usize) -> Fut,
        Fut: Future<Output = Result<T, LangChainError>>,
    {
        self.policy.run(operation).await
    }
}

impl AgentMiddleware for ToolRetryMiddleware {
    fn wrap_tool_call(
        &self,
        request: ToolCallRequest,
        handler: ToolCallHandler,
    ) -> BoxFuture<'static, Result<ToolMessage, LangChainError>> {
        let retry = self.clone();
        Box::pin(async move {
            retry
                .retry_tool_call(|_| {
                    let request = request.clone();
                    let handler = handler.clone();
                    async move { handler(request).await }
                })
                .await
        })
    }
}
