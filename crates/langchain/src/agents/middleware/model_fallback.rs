use std::sync::Arc;

use futures_util::future::BoxFuture;
use langchain_core::LangChainError;
use langchain_core::language_models::BaseChatModel;

use crate::agents::middleware::types::{
    AgentMiddleware, ModelCallHandler, ModelRequest, ModelResponse,
};
use crate::chat_models::init_chat_model;

#[derive(Clone)]
pub struct ModelFallbackMiddleware {
    models: Vec<Arc<dyn BaseChatModel>>,
}

impl ModelFallbackMiddleware {
    pub fn new(models: Vec<String>) -> Self {
        let resolved = models
            .into_iter()
            .filter_map(|model| init_chat_model(&model, None, None, None).ok())
            .map(Arc::<dyn BaseChatModel>::from)
            .collect();
        Self { models: resolved }
    }

    pub fn from_models(models: Vec<Arc<dyn BaseChatModel>>) -> Self {
        Self { models }
    }
}

impl AgentMiddleware for ModelFallbackMiddleware {
    fn wrap_model_call(
        &self,
        request: ModelRequest,
        handler: ModelCallHandler,
    ) -> BoxFuture<'static, Result<ModelResponse, LangChainError>> {
        let fallbacks = self.models.clone();
        Box::pin(async move {
            let mut last_error = match handler(request.clone()).await {
                Ok(response) => return Ok(response),
                Err(error) => error,
            };

            for model in fallbacks {
                match handler(request.clone().with_model(model)).await {
                    Ok(response) => return Ok(response),
                    Err(error) => last_error = error,
                }
            }

            Err(last_error)
        })
    }
}
