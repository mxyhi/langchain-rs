use thiserror::Error;

#[derive(Debug, Error)]
pub enum LangChainError {
    #[error("missing prompt variable `{name}`")]
    MissingPromptVariable { name: String },
    #[error("prompt variable `{name}` must be {expected}")]
    InvalidPromptValue {
        name: String,
        expected: &'static str,
    },
    #[error("template still contains unresolved variables: {template}")]
    UnresolvedTemplateVariables { template: String },
    #[error("chat model requires at least one message")]
    EmptyMessages,
    #[error("unsupported operation: {message}")]
    Unsupported { message: String },
    #[error("request failed: {message}")]
    Request { message: String },
    #[error("upstream returned http {status}: {body}")]
    HttpStatus { status: u16, body: String },
    #[error("json serialization failed: {0}")]
    Json(#[from] serde_json::Error),
}

impl LangChainError {
    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::Unsupported {
            message: message.into(),
        }
    }

    pub fn request(message: impl Into<String>) -> Self {
        Self::Request {
            message: message.into(),
        }
    }
}
