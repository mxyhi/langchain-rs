//! Legacy/classic compatibility surface.
//!
//! This crate mirrors the role of Python `langchain_classic`: it is the landing
//! zone for APIs that belong to the legacy/classic package rather than the new
//! facade crates. The implementation is intentionally minimal for now, but the
//! package boundary is real and ready to absorb classic-only modules later.

pub mod base_language;
pub mod base_memory;
pub mod chains;
pub mod example_generator;

pub mod agents {
    pub use langchain::agents::*;
}

pub mod cache {
    pub use langchain_core::caches::*;
}

pub mod callbacks {
    pub use langchain_core::callbacks::*;
}

pub mod chat_loaders {
    pub use langchain_core::chat_loaders::*;
}

pub mod chat_models {
    pub use langchain::chat_models::{
        ChatAnthropic, ChatDeepSeek, ChatFireworks, ChatGroq, ChatHuggingFace, ChatMistralAI,
        ChatOllama, ChatOpenAI, ChatOpenRouter, ChatPerplexity, ChatXAI, ConfigurableChatModel,
    };
    pub use langchain_core::language_models::{
        BaseChatModel, ParrotChatModel, StructuredOutput, StructuredOutputMethod,
        StructuredOutputOptions, StructuredOutputSchema, ToolBindingOptions, ToolChoice,
    };

    pub fn init_chat_model(
        model: &str,
    ) -> Result<Box<dyn BaseChatModel>, langchain_core::LangChainError> {
        langchain::chat_models::init_chat_model(model, None, None, None)
    }
}

pub mod documents {
    pub use langchain_core::documents::*;
}

pub mod embeddings {
    pub use langchain_core::embeddings::*;
}

pub mod env;

pub mod hub;

pub mod formatting;

pub mod globals {
    pub use langchain_core::globals::*;
}

pub mod input;

pub mod language_models {
    pub use langchain_core::language_models::*;
}

pub mod llms {
    pub use langchain_core::language_models::{BaseLLM, ParrotLLM};
}

pub mod memory;

pub mod messages {
    pub use langchain_core::messages::*;
}

pub mod model_laboratory;

pub mod output_parsers {
    pub use langchain_core::output_parsers::*;
}

pub mod python;

pub mod load {
    pub use langchain_core::load::*;
}

pub mod prompt_values {
    pub use langchain_core::prompt_values::*;
}

pub mod prompts;

pub mod example_selectors {
    pub use langchain_core::example_selectors::*;
}

pub mod retrievers {
    pub use langchain_core::retrievers::*;
}

pub mod requests;

pub mod runnables {
    pub use langchain_core::runnables::*;
}

pub mod serpapi;

pub mod text_splitter {
    pub use langchain_text_splitters::{
        CharacterTextSplitter, Language, RecursiveCharacterTextSplitter, TextSplitter,
        TokenTextSplitter, Tokenizer, split_text_on_tokens,
    };
}

pub mod sql_database;

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

pub mod storage {
    pub use langchain_core::stores::*;
}

pub mod document_loaders {
    pub use langchain_core::document_loaders::*;
}

pub mod indexing {
    pub use langchain_core::indexing::*;
}

pub mod utilities {
    pub use crate::python::PythonREPL;
    pub use crate::requests::{Requests, RequestsWrapper, TextRequestsWrapper};
    pub use crate::serpapi::SerpAPIWrapper;
    pub use crate::sql_database::SQLDatabase;
    pub use langchain_core::messages::{message_to_dict, messages_to_dict, trim_messages};
}

pub mod utils {
    pub mod formatting {
        pub use crate::formatting::*;
    }

    pub mod input {
        pub use crate::input::*;
    }
}

pub mod vectorstores {
    pub use langchain_core::vectorstores::*;
}

pub use chains::{ConversationChain, LLMChain};
pub use langchain_core::LangChainError;
pub use prompts::{Prompt, PromptTemplate};
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

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
