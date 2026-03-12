//! Legacy/classic compatibility surface.
//!
//! This crate mirrors the role of Python `langchain_classic`: it is the landing
//! zone for APIs that belong to the legacy/classic package rather than the new
//! facade crates. The implementation is intentionally minimal for now, but the
//! package boundary is real and ready to absorb classic-only modules later.

pub mod chains;

pub mod chat_models {
    pub use langchain_core::language_models::{
        BaseChatModel, ParrotChatModel, StructuredOutput, StructuredOutputMethod,
        StructuredOutputOptions, StructuredOutputSchema, ToolBindingOptions, ToolChoice,
    };

    pub fn init_chat_model(
        _model: &str,
    ) -> Result<Box<dyn BaseChatModel>, langchain_core::LangChainError> {
        Err(langchain_core::LangChainError::unsupported(
            "classic chat_models::init_chat_model is not implemented in this milestone",
        ))
    }
}

pub mod documents {
    pub use langchain_core::documents::*;
}

pub mod embeddings {
    pub use langchain_core::embeddings::*;
}

pub mod language_models {
    pub use langchain_core::language_models::*;
}

pub mod llms {
    pub use langchain_core::language_models::{BaseLLM, ParrotLLM};
}

pub mod messages {
    pub use langchain_core::messages::*;
}

pub mod output_parsers {
    pub use langchain_core::output_parsers::*;
}

pub mod prompt_values {
    pub use langchain_core::prompt_values::*;
}

pub mod prompts {
    pub use langchain_core::prompts::*;
}

pub mod retrievers {
    pub use langchain_core::retrievers::*;
}

pub mod runnables {
    pub use langchain_core::runnables::*;
}

pub mod tools {
    pub use langchain_core::tools::*;
}

pub mod schema {
    pub use langchain_core::documents::Document;
    pub use langchain_core::messages::*;
    pub use langchain_core::outputs::*;
    pub use langchain_core::prompt_values::*;
}

pub mod docstore {
    pub use langchain_core::documents::Document;
}

pub mod utilities {
    pub use langchain_core::messages::{message_to_dict, messages_to_dict, trim_messages};
}

pub mod vectorstores {
    pub use langchain_core::vectorstores::*;
}

pub use chains::{ConversationChain, LLMChain};
pub use langchain_core::LangChainError;

/// Marker type for the classic/legacy package boundary.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ClassicPackage;

impl ClassicPackage {
    /// Canonical crate name used in the Rust workspace.
    pub const fn package_name(self) -> &'static str {
        "langchain-classic"
    }

    /// Short explanation of why this crate exists.
    pub const fn purpose(self) -> &'static str {
        "Legacy compatibility surface corresponding to Python langchain_classic."
    }
}
