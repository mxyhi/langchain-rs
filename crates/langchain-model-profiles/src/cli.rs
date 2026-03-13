use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};

use crate::{ProviderCapabilities, ProviderProfile, provider, providers};

const MODELS_DEV_CATALOG_URL: &str = "https://models.dev/api.json";
const MODELS_DEV_CATALOG_URL_ENV: &str = "LANGCHAIN_MODEL_PROFILES_CATALOG_URL";
const GENERATED_PROFILES_FILENAME: &str = "_profiles.json";

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

    fn failure(message: impl Into<String>) -> Self {
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct RefreshArgs {
    provider: String,
    data_dir: PathBuf,
    catalog: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RefreshResult {
    provider: String,
    output_path: PathBuf,
    model_count: usize,
}

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
        "langchain-profiles\n\nUSAGE:\n  langchain-profiles list\n  langchain-profiles show <provider>\n  langchain-profiles provider <provider>\n  langchain-profiles capability <capability>\n  langchain-profiles refresh --provider <provider> --data-dir <path> [--catalog <path>]\n\nCAPABILITIES:\n  {capabilities}"
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
        [command, rest @ ..] if command == "refresh" => {
            let refresh = parse_refresh_args(rest)?;
            let result = refresh_profiles(&refresh)?;
            Ok(render_refresh_result(&result))
        }
        _ => Err(CliError::usage(format!(
            "invalid arguments\n\n{}",
            render_help()
        ))),
    }
}

fn parse_refresh_args(args: &[String]) -> Result<RefreshArgs, CliError> {
    let mut provider_key = None;
    let mut data_dir = None;
    let mut catalog = None;
    let mut index = 0;

    while index < args.len() {
        let flag = &args[index];
        let value = args.get(index + 1).ok_or_else(|| {
            CliError::usage(format!("missing value for {flag}\n\n{}", render_help()))
        })?;

        match flag.as_str() {
            "--provider" => provider_key = Some(value.clone()),
            "--data-dir" => data_dir = Some(PathBuf::from(value)),
            "--catalog" => catalog = Some(PathBuf::from(value)),
            _ => {
                return Err(CliError::usage(format!(
                    "unknown refresh argument: {flag}\n\n{}",
                    render_help()
                )));
            }
        }

        index += 2;
    }

    Ok(RefreshArgs {
        provider: provider_key.ok_or_else(|| {
            CliError::usage(format!("missing required --provider\n\n{}", render_help()))
        })?,
        data_dir: data_dir.ok_or_else(|| {
            CliError::usage(format!("missing required --data-dir\n\n{}", render_help()))
        })?,
        catalog,
    })
}

fn refresh_profiles(args: &RefreshArgs) -> Result<RefreshResult, CliError> {
    let profile = describe_provider(&args.provider)
        .ok_or_else(|| CliError::not_found(format!("provider not found: {}", args.provider)))?;
    let catalog = load_catalog(args.catalog.as_deref())?;
    let models = extract_models_for_provider(&catalog, profile)?;
    let (provider_overrides, model_overrides) = load_augmentations(&args.data_dir)?;

    let mut merged_models = BTreeMap::new();
    for (model_id, model_data) in models {
        let base = model_data_to_profile(model_data)?;
        let merged = apply_overrides(base, &provider_overrides, model_overrides.get(model_id));
        merged_models.insert(model_id.clone(), Value::Object(merged));
    }

    for (model_id, overrides) in &model_overrides {
        if merged_models.contains_key(model_id) {
            continue;
        }

        let merged = apply_overrides(Map::new(), &provider_overrides, Some(overrides));
        merged_models.insert(model_id.clone(), Value::Object(merged));
    }

    fs::create_dir_all(&args.data_dir).map_err(|error| {
        CliError::failure(format!(
            "failed to create data directory {}: {error}",
            args.data_dir.display()
        ))
    })?;

    let output_path = args.data_dir.join(GENERATED_PROFILES_FILENAME);
    let document = build_profiles_document(profile, merged_models);
    let encoded = serde_json::to_vec_pretty(&document).map_err(|error| {
        CliError::failure(format!("failed to serialize generated profiles: {error}"))
    })?;
    fs::write(&output_path, encoded).map_err(|error| {
        CliError::failure(format!(
            "failed to write generated profiles {}: {error}",
            output_path.display()
        ))
    })?;

    Ok(RefreshResult {
        provider: profile.key.to_owned(),
        output_path,
        model_count: document["models"]
            .as_object()
            .map_or(0, |models| models.len()),
    })
}

fn render_refresh_result(result: &RefreshResult) -> String {
    format!(
        "provider: {}\noutput: {}\nmodels: {}",
        result.provider,
        result.output_path.display(),
        result.model_count,
    )
}

fn load_catalog(source: Option<&Path>) -> Result<Value, CliError> {
    match source {
        Some(path) => {
            let contents = fs::read_to_string(path).map_err(|error| {
                CliError::failure(format!(
                    "failed to read catalog {}: {error}",
                    path.display()
                ))
            })?;
            serde_json::from_str(&contents).map_err(|error| {
                CliError::failure(format!(
                    "failed to parse catalog JSON {}: {error}",
                    path.display()
                ))
            })
        }
        None => {
            let catalog_url = models_dev_catalog_url();
            let runtime = tokio::runtime::Runtime::new().map_err(|error| {
                CliError::failure(format!("failed to start runtime for refresh: {error}"))
            })?;
            runtime.block_on(async {
                let response = reqwest::get(&catalog_url).await.map_err(|error| {
                    CliError::failure(format!(
                        "failed to fetch models.dev catalog {catalog_url}: {error}"
                    ))
                })?;
                let response = response.error_for_status().map_err(|error| {
                    CliError::failure(format!(
                        "models.dev catalog request failed {catalog_url}: {error}"
                    ))
                })?;
                response.json::<Value>().await.map_err(|error| {
                    CliError::failure(format!(
                        "failed to decode models.dev catalog {catalog_url}: {error}"
                    ))
                })
            })
        }
    }
}

fn models_dev_catalog_url() -> String {
    std::env::var(MODELS_DEV_CATALOG_URL_ENV)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| MODELS_DEV_CATALOG_URL.to_owned())
}

fn extract_models_for_provider<'a>(
    catalog: &'a Value,
    profile: &ProviderProfile,
) -> Result<&'a Map<String, Value>, CliError> {
    let providers = catalog.as_object().ok_or_else(|| {
        CliError::failure("invalid catalog format: expected top-level JSON object")
    })?;
    let provider_data = providers.get(profile.key).ok_or_else(|| {
        CliError::not_found(format!("provider not found in catalog: {}", profile.key))
    })?;
    let models = provider_data
        .get("models")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            CliError::failure(format!(
                "invalid catalog format for provider {}: missing models object",
                profile.key
            ))
        })?;

    Ok(models)
}

fn load_augmentations(
    data_dir: &Path,
) -> Result<(Map<String, Value>, BTreeMap<String, Map<String, Value>>), CliError> {
    let path = data_dir.join("profile_augmentations.toml");
    if !path.exists() {
        return Ok((Map::new(), BTreeMap::new()));
    }

    let contents = fs::read_to_string(&path).map_err(|error| {
        CliError::failure(format!(
            "failed to read augmentations {}: {error}",
            path.display()
        ))
    })?;
    let mut provider_overrides = Map::new();
    let mut model_overrides: BTreeMap<String, Map<String, Value>> = BTreeMap::new();

    let mut section = AugmentationSection::Ignore;
    for raw_line in contents.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if let Some(next_section) = parse_augmentation_section(line)? {
            section = next_section;
            continue;
        }

        let (key, value) = parse_assignment(line)?;
        match &section {
            AugmentationSection::Provider => {
                provider_overrides.insert(key.to_owned(), parse_scalar_value(value)?);
            }
            AugmentationSection::Model(model_id) => {
                model_overrides
                    .entry(model_id.clone())
                    .or_default()
                    .insert(key.to_owned(), parse_scalar_value(value)?);
            }
            AugmentationSection::Ignore => {}
        }
    }

    Ok((provider_overrides, model_overrides))
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum AugmentationSection {
    Ignore,
    Provider,
    Model(String),
}

fn parse_augmentation_section(line: &str) -> Result<Option<AugmentationSection>, CliError> {
    if !(line.starts_with('[') && line.ends_with(']')) {
        return Ok(None);
    }

    let section = &line[1..line.len() - 1];
    if section == "overrides" {
        return Ok(Some(AugmentationSection::Provider));
    }

    if let Some(model_id) = section
        .strip_prefix("overrides.\"")
        .and_then(|value| value.strip_suffix('"'))
    {
        return Ok(Some(AugmentationSection::Model(model_id.to_owned())));
    }

    if section.starts_with("overrides.") {
        return Err(CliError::failure(format!(
            "unsupported augmentations section: {section}"
        )));
    }

    Ok(Some(AugmentationSection::Ignore))
}

fn parse_assignment(line: &str) -> Result<(&str, &str), CliError> {
    let (key, value) = line
        .split_once('=')
        .ok_or_else(|| CliError::failure(format!("invalid augmentation line: {line}")))?;
    Ok((key.trim(), value.trim()))
}

fn parse_scalar_value(raw: &str) -> Result<Value, CliError> {
    if raw.eq_ignore_ascii_case("true") {
        return Ok(Value::Bool(true));
    }
    if raw.eq_ignore_ascii_case("false") {
        return Ok(Value::Bool(false));
    }
    if raw.starts_with('"') && raw.ends_with('"') && raw.len() >= 2 {
        return Ok(Value::String(raw[1..raw.len() - 1].to_owned()));
    }
    if let Ok(value) = raw.parse::<i64>() {
        return Ok(json!(value));
    }
    if let Ok(value) = raw.parse::<f64>() {
        return Ok(json!(value));
    }

    Err(CliError::failure(format!(
        "unsupported augmentation value: {raw}"
    )))
}

fn model_data_to_profile(model_data: &Value) -> Result<Map<String, Value>, CliError> {
    let object = model_data.as_object().ok_or_else(|| {
        CliError::failure("invalid model data: expected a JSON object for each model")
    })?;
    let limit = object.get("limit").and_then(Value::as_object);
    let modalities = object.get("modalities").and_then(Value::as_object);
    let input_modalities = modalities
        .and_then(|modalities| modalities.get("input"))
        .and_then(Value::as_array);
    let output_modalities = modalities
        .and_then(|modalities| modalities.get("output"))
        .and_then(Value::as_array);

    let mut profile = Map::new();
    insert_optional(
        &mut profile,
        "max_input_tokens",
        limit.and_then(|value| value.get("context")),
    );
    insert_optional(
        &mut profile,
        "max_output_tokens",
        limit.and_then(|value| value.get("output")),
    );
    insert_bool(
        &mut profile,
        "text_inputs",
        contains_modality(input_modalities, "text"),
    );
    insert_bool(
        &mut profile,
        "image_inputs",
        contains_modality(input_modalities, "image"),
    );
    insert_bool(
        &mut profile,
        "audio_inputs",
        contains_modality(input_modalities, "audio"),
    );
    insert_bool(
        &mut profile,
        "pdf_inputs",
        contains_modality(input_modalities, "pdf") || as_bool(object.get("pdf_inputs")),
    );
    insert_bool(
        &mut profile,
        "video_inputs",
        contains_modality(input_modalities, "video"),
    );
    insert_bool(
        &mut profile,
        "text_outputs",
        contains_modality(output_modalities, "text"),
    );
    insert_bool(
        &mut profile,
        "image_outputs",
        contains_modality(output_modalities, "image"),
    );
    insert_bool(
        &mut profile,
        "audio_outputs",
        contains_modality(output_modalities, "audio"),
    );
    insert_bool(
        &mut profile,
        "video_outputs",
        contains_modality(output_modalities, "video"),
    );
    insert_optional(&mut profile, "reasoning_output", object.get("reasoning"));
    insert_optional(&mut profile, "tool_calling", object.get("tool_call"));
    insert_optional(&mut profile, "tool_choice", object.get("tool_choice"));
    insert_optional(
        &mut profile,
        "structured_output",
        object.get("structured_output"),
    );
    insert_optional(
        &mut profile,
        "image_url_inputs",
        object.get("image_url_inputs"),
    );
    insert_optional(
        &mut profile,
        "image_tool_message",
        object.get("image_tool_message"),
    );
    insert_optional(
        &mut profile,
        "pdf_tool_message",
        object.get("pdf_tool_message"),
    );

    Ok(profile)
}

fn apply_overrides(
    mut base: Map<String, Value>,
    provider_overrides: &Map<String, Value>,
    model_overrides: Option<&Map<String, Value>>,
) -> Map<String, Value> {
    merge_json_map(&mut base, provider_overrides);
    if let Some(model_overrides) = model_overrides {
        merge_json_map(&mut base, model_overrides);
    }
    base
}

fn merge_json_map(target: &mut Map<String, Value>, overrides: &Map<String, Value>) {
    for (key, value) in overrides {
        target.insert(key.clone(), value.clone());
    }
}

fn build_profiles_document(profile: &ProviderProfile, models: BTreeMap<String, Value>) -> Value {
    json!({
        "provider": profile.key,
        "package_name": profile.package_name,
        "default_base_url": profile.default_base_url,
        "chat_model_prefixes": profile.chat_model_prefixes,
        "capabilities": {
            "chat_model": profile.capabilities.chat_model,
            "llm": profile.capabilities.llm,
            "embeddings": profile.capabilities.embeddings,
            "vector_store": profile.capabilities.vector_store,
            "retriever": profile.capabilities.retriever,
            "parser_or_tooling": profile.capabilities.parser_or_tooling
        },
        "exports": profile.exports,
        "models": models
    })
}

fn contains_modality(values: Option<&Vec<Value>>, expected: &str) -> bool {
    values.is_some_and(|values| {
        values
            .iter()
            .any(|value| value.as_str().is_some_and(|actual| actual == expected))
    })
}

fn as_bool(value: Option<&Value>) -> bool {
    value.and_then(Value::as_bool).unwrap_or(false)
}

fn insert_optional(target: &mut Map<String, Value>, key: &str, value: Option<&Value>) {
    if let Some(value) = value.filter(|value| !value.is_null()) {
        target.insert(key.to_owned(), value.clone());
    }
}

fn insert_bool(target: &mut Map<String, Value>, key: &str, value: bool) {
    target.insert(key.to_owned(), Value::Bool(value));
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
