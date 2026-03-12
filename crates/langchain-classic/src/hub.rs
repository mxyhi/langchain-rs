use langchain_core::LangChainError;
use langchain_core::prompts::PromptTemplate;

pub fn push<T>(_repo_full_name: &str, _object: &T) -> Result<String, LangChainError> {
    Err(LangChainError::unsupported(
        "langchain hub push is not implemented in this Rust workspace",
    ))
}

pub fn pull(_owner_repo_commit: &str) -> Result<PromptTemplate, LangChainError> {
    Err(LangChainError::unsupported(
        "langchain hub pull is not implemented in this Rust workspace",
    ))
}
