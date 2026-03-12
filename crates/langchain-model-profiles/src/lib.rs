pub mod cli;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ProviderCapabilities {
    pub chat_model: bool,
    pub llm: bool,
    pub embeddings: bool,
    pub vector_store: bool,
    pub retriever: bool,
    pub parser_or_tooling: bool,
}

impl ProviderCapabilities {
    pub const fn new() -> Self {
        Self {
            chat_model: false,
            llm: false,
            embeddings: false,
            vector_store: false,
            retriever: false,
            parser_or_tooling: false,
        }
    }

    pub const fn with_chat_model(mut self) -> Self {
        self.chat_model = true;
        self
    }

    pub const fn with_llm(mut self) -> Self {
        self.llm = true;
        self
    }

    pub const fn with_embeddings(mut self) -> Self {
        self.embeddings = true;
        self
    }

    pub const fn with_vector_store(mut self) -> Self {
        self.vector_store = true;
        self
    }

    pub const fn with_retriever(mut self) -> Self {
        self.retriever = true;
        self
    }

    pub const fn with_parser_or_tooling(mut self) -> Self {
        self.parser_or_tooling = true;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderProfile {
    pub key: &'static str,
    pub package_name: &'static str,
    pub default_base_url: Option<&'static str>,
    pub chat_model_prefixes: &'static [&'static str],
    pub capabilities: ProviderCapabilities,
    pub exports: &'static [&'static str],
}

impl ProviderProfile {
    pub const fn supports_chat_model(self) -> bool {
        self.capabilities.chat_model
    }

    pub const fn supports_llm(self) -> bool {
        self.capabilities.llm
    }

    pub const fn supports_embeddings(self) -> bool {
        self.capabilities.embeddings
    }
}

const OPENAI_CAPABILITIES: ProviderCapabilities = ProviderCapabilities::new()
    .with_chat_model()
    .with_llm()
    .with_embeddings()
    .with_parser_or_tooling();
const ANTHROPIC_CAPABILITIES: ProviderCapabilities = ProviderCapabilities::new()
    .with_chat_model()
    .with_llm()
    .with_parser_or_tooling();
const OLLAMA_CAPABILITIES: ProviderCapabilities = ProviderCapabilities::new()
    .with_chat_model()
    .with_llm()
    .with_embeddings();
const CHAT_ONLY_CAPABILITIES: ProviderCapabilities = ProviderCapabilities::new().with_chat_model();
const FIREWORKS_CAPABILITIES: ProviderCapabilities = ProviderCapabilities::new()
    .with_chat_model()
    .with_llm()
    .with_embeddings();
const HUGGINGFACE_CAPABILITIES: ProviderCapabilities = ProviderCapabilities::new()
    .with_chat_model()
    .with_llm()
    .with_embeddings();
const MISTRAL_CAPABILITIES: ProviderCapabilities = ProviderCapabilities::new()
    .with_chat_model()
    .with_embeddings();
const NOMIC_CAPABILITIES: ProviderCapabilities = ProviderCapabilities::new().with_embeddings();
const EXA_CAPABILITIES: ProviderCapabilities = ProviderCapabilities::new()
    .with_retriever()
    .with_parser_or_tooling();
const QDRANT_CAPABILITIES: ProviderCapabilities = ProviderCapabilities::new()
    .with_vector_store()
    .with_embeddings();
const CHROMA_CAPABILITIES: ProviderCapabilities = ProviderCapabilities::new().with_vector_store();
const PERPLEXITY_CAPABILITIES: ProviderCapabilities = ProviderCapabilities::new()
    .with_chat_model()
    .with_retriever()
    .with_parser_or_tooling();

pub const PROVIDERS: &[ProviderProfile] = &[
    ProviderProfile {
        key: "openai",
        package_name: "langchain-openai",
        default_base_url: Some("https://api.openai.com/v1"),
        chat_model_prefixes: &["gpt-", "o1", "o3", "chatgpt", "text-davinci"],
        capabilities: OPENAI_CAPABILITIES,
        exports: &[
            "ChatOpenAI",
            "OpenAIEmbeddings",
            "OpenAI",
            "AzureChatOpenAI",
            "AzureOpenAIEmbeddings",
            "AzureOpenAI",
            "custom_tool",
        ],
    },
    ProviderProfile {
        key: "anthropic",
        package_name: "langchain-anthropic",
        default_base_url: Some("https://api.anthropic.com"),
        chat_model_prefixes: &["claude"],
        capabilities: ANTHROPIC_CAPABILITIES,
        exports: &["ChatAnthropic", "AnthropicLLM", "convert_to_anthropic_tool"],
    },
    ProviderProfile {
        key: "ollama",
        package_name: "langchain-ollama",
        default_base_url: Some("http://localhost:11434/v1"),
        chat_model_prefixes: &[],
        capabilities: OLLAMA_CAPABILITIES,
        exports: &["ChatOllama", "OllamaEmbeddings", "OllamaLLM"],
    },
    ProviderProfile {
        key: "deepseek",
        package_name: "langchain-deepseek",
        default_base_url: Some("https://api.deepseek.com/v1"),
        chat_model_prefixes: &["deepseek"],
        capabilities: CHAT_ONLY_CAPABILITIES,
        exports: &["ChatDeepSeek"],
    },
    ProviderProfile {
        key: "fireworks",
        package_name: "langchain-fireworks",
        default_base_url: Some("https://api.fireworks.ai/inference/v1"),
        chat_model_prefixes: &["accounts/fireworks"],
        capabilities: FIREWORKS_CAPABILITIES,
        exports: &["ChatFireworks", "FireworksEmbeddings", "Fireworks"],
    },
    ProviderProfile {
        key: "groq",
        package_name: "langchain-groq",
        default_base_url: Some("https://api.groq.com/openai/v1"),
        chat_model_prefixes: &[],
        capabilities: CHAT_ONLY_CAPABILITIES,
        exports: &["ChatGroq"],
    },
    ProviderProfile {
        key: "huggingface",
        package_name: "langchain-huggingface",
        default_base_url: None,
        chat_model_prefixes: &[],
        capabilities: HUGGINGFACE_CAPABILITIES,
        exports: &[
            "ChatHuggingFace",
            "HuggingFaceEmbeddings",
            "HuggingFaceEndpointEmbeddings",
            "HuggingFaceEndpoint",
            "HuggingFacePipeline",
        ],
    },
    ProviderProfile {
        key: "mistralai",
        package_name: "langchain-mistralai",
        default_base_url: Some("https://api.mistral.ai/v1"),
        chat_model_prefixes: &["mistral"],
        capabilities: MISTRAL_CAPABILITIES,
        exports: &["ChatMistralAI", "MistralAIEmbeddings"],
    },
    ProviderProfile {
        key: "nomic",
        package_name: "langchain-nomic",
        default_base_url: Some("https://api-atlas.nomic.ai"),
        chat_model_prefixes: &[],
        capabilities: NOMIC_CAPABILITIES,
        exports: &["NomicEmbeddings"],
    },
    ProviderProfile {
        key: "openrouter",
        package_name: "langchain-openrouter",
        default_base_url: Some("https://openrouter.ai/api/v1"),
        chat_model_prefixes: &[],
        capabilities: CHAT_ONLY_CAPABILITIES,
        exports: &["ChatOpenRouter"],
    },
    ProviderProfile {
        key: "perplexity",
        package_name: "langchain-perplexity",
        default_base_url: Some("https://api.perplexity.ai"),
        chat_model_prefixes: &["sonar"],
        capabilities: PERPLEXITY_CAPABILITIES,
        exports: &[
            "ChatPerplexity",
            "PerplexitySearchRetriever",
            "PerplexitySearchResults",
            "UserLocation",
            "WebSearchOptions",
            "MediaResponse",
            "MediaResponseOverrides",
            "ReasoningJsonOutputParser",
            "ReasoningStructuredOutputParser",
            "strip_think_tags",
        ],
    },
    ProviderProfile {
        key: "qdrant",
        package_name: "langchain-qdrant",
        default_base_url: Some("http://localhost:6333"),
        chat_model_prefixes: &[],
        capabilities: QDRANT_CAPABILITIES,
        exports: &[
            "FastEmbedSparse",
            "Qdrant",
            "QdrantVectorStore",
            "RetrievalMode",
            "SparseEmbeddings",
            "SparseVector",
        ],
    },
    ProviderProfile {
        key: "chroma",
        package_name: "langchain-chroma",
        default_base_url: Some("http://localhost:8000"),
        chat_model_prefixes: &[],
        capabilities: CHROMA_CAPABILITIES,
        exports: &["Chroma"],
    },
    ProviderProfile {
        key: "xai",
        package_name: "langchain-xai",
        default_base_url: Some("https://api.x.ai/v1"),
        chat_model_prefixes: &["grok"],
        capabilities: CHAT_ONLY_CAPABILITIES,
        exports: &["ChatXAI"],
    },
    ProviderProfile {
        key: "exa",
        package_name: "langchain-exa",
        default_base_url: Some("https://api.exa.ai"),
        chat_model_prefixes: &[],
        capabilities: EXA_CAPABILITIES,
        exports: &[
            "ExaSearchRetriever",
            "ExaSearchResults",
            "ExaFindSimilarResults",
        ],
    },
];

pub fn providers() -> &'static [ProviderProfile] {
    PROVIDERS
}

pub fn provider(key: &str) -> Option<&'static ProviderProfile> {
    let normalized = normalize_provider_key(key);
    PROVIDERS.iter().find(|profile| profile.key == normalized)
}

pub fn supported_chat_providers() -> Vec<&'static str> {
    PROVIDERS
        .iter()
        .filter(|profile| profile.supports_chat_model())
        .map(|profile| profile.key)
        .collect()
}

pub fn supported_embedding_providers() -> Vec<&'static str> {
    PROVIDERS
        .iter()
        .filter(|profile| profile.supports_embeddings())
        .map(|profile| profile.key)
        .collect()
}

pub fn infer_chat_provider(model_name: &str) -> Option<&'static ProviderProfile> {
    let normalized = model_name.trim().to_ascii_lowercase();
    PROVIDERS.iter().find(|profile| {
        profile.supports_chat_model()
            && profile
                .chat_model_prefixes
                .iter()
                .any(|prefix| normalized.starts_with(prefix))
    })
}

pub fn normalize_provider_key(provider: &str) -> String {
    provider.replace('-', "_").trim().to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::{
        infer_chat_provider, normalize_provider_key, provider, supported_chat_providers,
        supported_embedding_providers,
    };

    #[test]
    fn supports_known_provider_lookup() {
        let profile = provider("openrouter").expect("provider should exist");
        assert_eq!(profile.package_name, "langchain-openrouter");
        assert!(profile.capabilities.chat_model);
        assert_eq!(
            profile.default_base_url,
            Some("https://openrouter.ai/api/v1")
        );
    }

    #[test]
    fn normalizes_provider_keys_for_case_and_whitespace() {
        assert_eq!(normalize_provider_key(" OpenAI "), "openai");
    }

    #[test]
    fn infers_provider_from_reference_prefixes() {
        assert_eq!(
            infer_chat_provider("claude-3-7-sonnet")
                .expect("anthropic prefix should infer")
                .key,
            "anthropic"
        );
        assert_eq!(
            infer_chat_provider("accounts/fireworks/models/llama-v3p1-8b-instruct")
                .expect("fireworks prefix should infer")
                .key,
            "fireworks"
        );
        assert_eq!(
            infer_chat_provider("grok-3-mini")
                .expect("xai prefix should infer")
                .key,
            "xai"
        );
    }

    #[test]
    fn returns_supported_provider_lists_by_capability() {
        let chat = supported_chat_providers();
        let embeddings = supported_embedding_providers();

        assert!(chat.contains(&"openai"));
        assert!(chat.contains(&"anthropic"));
        assert!(chat.contains(&"huggingface"));
        assert!(chat.contains(&"perplexity"));
        assert!(embeddings.contains(&"openai"));
        assert!(embeddings.contains(&"mistralai"));
        assert!(embeddings.contains(&"huggingface"));
        assert!(embeddings.contains(&"nomic"));
        assert!(!embeddings.contains(&"anthropic"));
    }

    #[test]
    fn perplexity_exports_match_boundary_surface() {
        let profile = provider("perplexity").expect("provider should exist");

        assert!(profile.exports.contains(&"ChatPerplexity"));
        assert!(profile.exports.contains(&"PerplexitySearchRetriever"));
        assert!(profile.exports.contains(&"PerplexitySearchResults"));
        assert!(profile.exports.contains(&"UserLocation"));
        assert!(profile.exports.contains(&"WebSearchOptions"));
        assert!(profile.exports.contains(&"MediaResponse"));
        assert!(profile.exports.contains(&"MediaResponseOverrides"));
    }
}
