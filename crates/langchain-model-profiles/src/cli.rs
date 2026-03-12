use std::error::Error;
use std::fmt;

use crate::{ProviderCapabilities, ProviderProfile, provider, providers};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Capability {
    ChatModel,
    Llm,
    Embeddings,
    VectorStore,
    Retriever,
    ParserOrTooling,
}

impl Capability {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ChatModel => "chat_model",
            Self::Llm => "llm",
            Self::Embeddings => "embeddings",
            Self::VectorStore => "vector_store",
            Self::Retriever => "retriever",
            Self::ParserOrTooling => "parser_or_tooling",
        }
    }

    pub const fn all() -> [Self; 6] {
        [
            Self::ChatModel,
            Self::Llm,
            Self::Embeddings,
            Self::VectorStore,
            Self::Retriever,
            Self::ParserOrTooling,
        ]
    }

    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "chat_model" | "chat-model" | "chat" => Some(Self::ChatModel),
            "llm" => Some(Self::Llm),
            "embeddings" | "embedding" => Some(Self::Embeddings),
            "vector_store" | "vector-store" | "vectorstore" => Some(Self::VectorStore),
            "retriever" | "retrievers" => Some(Self::Retriever),
            "parser_or_tooling" | "parser-tooling" | "parser" | "tooling" => {
                Some(Self::ParserOrTooling)
            }
            _ => None,
        }
    }

    const fn is_supported(self, capabilities: ProviderCapabilities) -> bool {
        match self {
            Self::ChatModel => capabilities.chat_model,
            Self::Llm => capabilities.llm,
            Self::Embeddings => capabilities.embeddings,
            Self::VectorStore => capabilities.vector_store,
            Self::Retriever => capabilities.retriever,
            Self::ParserOrTooling => capabilities.parser_or_tooling,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CliError {
    message: String,
    exit_code: i32,
}

impl CliError {
    fn usage(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            exit_code: 2,
        }
    }

    fn not_found(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            exit_code: 1,
        }
    }

    pub const fn exit_code(&self) -> i32 {
        self.exit_code
    }
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for CliError {}

pub fn describe_provider(key: &str) -> Option<&'static ProviderProfile> {
    provider(key)
}

pub fn render_provider_table() -> String {
    let mut lines = vec![format!(
        "{:<14} {:<28} {}",
        "provider", "package", "capabilities"
    )];

    for profile in providers() {
        lines.push(format!(
            "{:<14} {:<28} {}",
            profile.key,
            profile.package_name,
            format_capabilities(profile.capabilities)
        ));
    }

    lines.join("\n")
}

pub fn render_provider_detail(key: &str) -> Result<String, CliError> {
    let profile = describe_provider(key)
        .ok_or_else(|| CliError::not_found(format!("unknown provider: {key}")))?;

    let prefixes = render_list(profile.chat_model_prefixes);
    let exports = render_exports(profile.exports);

    Ok(format!(
        "provider: {provider}\npackage: {package}\ndefault_base_url: {base_url}\nchat_model_prefixes: {prefixes}\ncapabilities:\n  chat_model: {chat_model}\n  llm: {llm}\n  embeddings: {embeddings}\n  vector_store: {vector_store}\n  retriever: {retriever}\n  parser_or_tooling: {parser_or_tooling}\nexports:\n{exports}",
        provider = profile.key,
        package = profile.package_name,
        base_url = profile.default_base_url.unwrap_or("(none)"),
        prefixes = prefixes,
        chat_model = yes_no(profile.capabilities.chat_model),
        llm = yes_no(profile.capabilities.llm),
        embeddings = yes_no(profile.capabilities.embeddings),
        vector_store = yes_no(profile.capabilities.vector_store),
        retriever = yes_no(profile.capabilities.retriever),
        parser_or_tooling = yes_no(profile.capabilities.parser_or_tooling),
        exports = exports,
    ))
}

pub fn render_capability_table(capability: &str) -> Result<String, CliError> {
    let capability = Capability::parse(capability).ok_or_else(|| {
        CliError::usage(format!(
            "unknown capability: {capability}\n\n{}",
            render_help()
        ))
    })?;

    let mut lines = vec![
        format!("capability: {}", capability.as_str()),
        format!("{:<14} {}", "provider", "package"),
    ];

    for profile in providers()
        .iter()
        .filter(|profile| capability.is_supported(profile.capabilities))
    {
        lines.push(format!("{:<14} {}", profile.key, profile.package_name));
    }

    Ok(lines.join("\n"))
}

pub fn render_help() -> String {
    let capabilities = Capability::all()
        .into_iter()
        .map(Capability::as_str)
        .collect::<Vec<_>>()
        .join(", ");

    format!(
        "langchain-profiles\n\nUSAGE:\n  langchain-profiles list\n  langchain-profiles show <provider>\n  langchain-profiles provider <provider>\n  langchain-profiles capability <capability>\n\nCAPABILITIES:\n  {capabilities}"
    )
}

pub fn run<I, S>(args: I) -> Result<String, CliError>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let args = args.into_iter().map(Into::into).collect::<Vec<_>>();

    match args.as_slice() {
        [] => Ok(render_help()),
        [flag] if is_help_flag(flag) => Ok(render_help()),
        [command] if command == "list" => Ok(render_provider_table()),
        [command, provider] if command == "show" || command == "provider" => {
            render_provider_detail(provider)
        }
        [command, capability] if command == "capability" => render_capability_table(capability),
        _ => Err(CliError::usage(format!(
            "invalid arguments\n\n{}",
            render_help()
        ))),
    }
}

fn format_capabilities(capabilities: ProviderCapabilities) -> String {
    Capability::all()
        .into_iter()
        .filter(|capability| capability.is_supported(capabilities))
        .map(Capability::as_str)
        .collect::<Vec<_>>()
        .join(",")
        .if_empty_then("none")
}

fn render_list(values: &[&str]) -> String {
    values.join(",").if_empty_then("(none)")
}

fn render_exports(exports: &[&str]) -> String {
    if exports.is_empty() {
        return "  - (none)".to_string();
    }

    exports
        .iter()
        .map(|export| format!("  - {export}"))
        .collect::<Vec<_>>()
        .join("\n")
}

const fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}

fn is_help_flag(flag: &str) -> bool {
    matches!(flag, "-h" | "--help" | "help")
}

trait IfEmptyThen {
    fn if_empty_then(self, fallback: &str) -> String;
}

impl IfEmptyThen for String {
    fn if_empty_then(self, fallback: &str) -> String {
        if self.is_empty() {
            fallback.to_string()
        } else {
            self
        }
    }
}
