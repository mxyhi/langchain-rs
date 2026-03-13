use std::thread;

use langchain_core::LangChainError;
use langchain_core::prompts::{PromptMetadata, PromptTemplate};
use reqwest::{Client, Url};
use serde::{Deserialize, Serialize};

const DEFAULT_HUB_API_URL: &str = "https://smith.langchain.com/hub";
const HUB_API_URL_ENV: &str = "LANGCHAIN_HUB_API_URL";
const HUB_API_KEY_ENV: &str = "LANGCHAIN_HUB_API_KEY";

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HubOptions {
    api_url: Option<String>,
    api_key: Option<String>,
}

impl HubOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_api_url(mut self, api_url: impl Into<String>) -> Self {
        self.api_url = Some(api_url.into());
        self
    }

    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    fn resolved_api_url(&self) -> String {
        self.api_url
            .clone()
            .or_else(|| std::env::var(HUB_API_URL_ENV).ok())
            .unwrap_or_else(|| DEFAULT_HUB_API_URL.to_owned())
    }

    fn resolved_api_key(&self) -> Option<String> {
        self.api_key
            .clone()
            .or_else(|| std::env::var(HUB_API_KEY_ENV).ok())
    }
}

#[derive(Debug, Clone)]
pub struct HubClient {
    http: Client,
    base_url: Url,
    api_key: Option<String>,
}

impl HubClient {
    pub fn from_env() -> Result<Self, LangChainError> {
        Self::from_options(&HubOptions::new())
    }

    pub fn from_options(options: &HubOptions) -> Result<Self, LangChainError> {
        let base_url = Url::parse(&options.resolved_api_url())
            .map_err(|error| LangChainError::request(format!("invalid hub api url: {error}")))?;

        Ok(Self {
            http: Client::new(),
            base_url,
            api_key: options.resolved_api_key(),
        })
    }

    pub fn push_prompt(
        &self,
        repo_full_name: &str,
        prompt: &PromptTemplate,
    ) -> Result<String, LangChainError> {
        let url = self.repo_url(repo_full_name)?;
        let manifest = HubPromptManifest::from_prompt_template(prompt)?;
        let request = HubPushRequest { manifest };
        let http = self.http.clone();
        let api_key = self.api_key.clone();

        run_http(async move {
            let mut builder = http.post(url).json(&request);
            if let Some(api_key) = api_key {
                builder = builder.header("x-api-key", api_key);
            }

            let response = builder.send().await.map_err(http_error)?;
            let body = parse_success_json::<HubPushResponse>(response).await?;
            Ok(body.url)
        })
    }

    pub fn pull_prompt(&self, owner_repo_commit: &str) -> Result<PromptTemplate, LangChainError> {
        let (repo_full_name, commit) = parse_owner_repo_commit(owner_repo_commit)?;
        let mut url = self.repo_url(&repo_full_name)?;
        if let Some(commit) = commit {
            url.query_pairs_mut().append_pair("commit", &commit);
        }

        let http = self.http.clone();
        let api_key = self.api_key.clone();

        run_http(async move {
            let mut builder = http.get(url);
            if let Some(api_key) = api_key {
                builder = builder.header("x-api-key", api_key);
            }

            let response = builder.send().await.map_err(http_error)?;
            let body = parse_success_json::<HubPullResponse>(response).await?;
            body.into_prompt_template()
        })
    }

    fn repo_url(&self, repo_full_name: &str) -> Result<Url, LangChainError> {
        let segments = repo_full_name
            .split('/')
            .map(str::trim)
            .filter(|segment| !segment.is_empty())
            .collect::<Vec<_>>();
        if segments.is_empty() {
            return Err(LangChainError::request(
                "hub repo name must contain at least one non-empty path segment",
            ));
        }

        let mut url = self.base_url.clone();
        {
            let mut path_segments = url
                .path_segments_mut()
                .map_err(|_| LangChainError::request("hub api url cannot be used as a base URL"))?;
            path_segments.pop_if_empty();
            path_segments.push("repos");
            for segment in &segments {
                path_segments.push(segment);
            }
        }
        Ok(url)
    }
}

pub fn push(repo_full_name: &str, prompt: &PromptTemplate) -> Result<String, LangChainError> {
    HubClient::from_env()?.push_prompt(repo_full_name, prompt)
}

pub fn push_with_options(
    repo_full_name: &str,
    prompt: &PromptTemplate,
    options: &HubOptions,
) -> Result<String, LangChainError> {
    HubClient::from_options(options)?.push_prompt(repo_full_name, prompt)
}

pub fn pull(owner_repo_commit: &str) -> Result<PromptTemplate, LangChainError> {
    HubClient::from_env()?.pull_prompt(owner_repo_commit)
}

pub fn pull_with_options(
    owner_repo_commit: &str,
    options: &HubOptions,
) -> Result<PromptTemplate, LangChainError> {
    HubClient::from_options(options)?.pull_prompt(owner_repo_commit)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct HubPromptManifest {
    kind: String,
    template: String,
}

impl HubPromptManifest {
    fn from_prompt_template(prompt: &PromptTemplate) -> Result<Self, LangChainError> {
        Ok(Self {
            kind: "prompt_template".to_owned(),
            template: prompt.template().to_owned(),
        })
    }

    fn into_prompt_template(
        self,
        metadata: PromptMetadata,
    ) -> Result<PromptTemplate, LangChainError> {
        if self.kind != "prompt_template" {
            return Err(LangChainError::unsupported(format!(
                "unsupported hub manifest kind `{}`; only prompt_template is supported",
                self.kind
            )));
        }
        Ok(PromptTemplate::new(self.template).with_metadata(metadata))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct HubPushRequest {
    manifest: HubPromptManifest,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct HubPushResponse {
    url: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct HubPullResponse {
    manifest: HubPromptManifest,
    #[serde(default)]
    owner: Option<String>,
    #[serde(default)]
    repo: Option<String>,
    #[serde(default)]
    commit_hash: Option<String>,
}

fn parse_owner_repo_commit(
    owner_repo_commit: &str,
) -> Result<(String, Option<String>), LangChainError> {
    let trimmed = owner_repo_commit.trim();
    if trimmed.is_empty() {
        return Err(LangChainError::request(
            "hub pull requires a non-empty owner/repo identifier",
        ));
    }

    let (repo_full_name, commit) = match trimmed.split_once(':') {
        Some((repo_full_name, commit)) if !commit.trim().is_empty() => (
            repo_full_name.trim().to_owned(),
            Some(commit.trim().to_owned()),
        ),
        Some((_repo_full_name, _)) => {
            return Err(LangChainError::request(
                "hub pull commit hash cannot be empty when `:` is present",
            ));
        }
        None => (trimmed.to_owned(), None),
    };

    Ok((repo_full_name, commit))
}

fn run_http<T, F>(future: F) -> Result<T, LangChainError>
where
    T: Send + 'static,
    F: std::future::Future<Output = Result<T, LangChainError>> + Send + 'static,
{
    thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|error| {
                LangChainError::request(format!("failed to build hub runtime: {error}"))
            })?;
        runtime.block_on(future)
    })
    .join()
    .map_err(|_| LangChainError::request("hub request thread panicked"))?
}

async fn parse_success_json<T>(response: reqwest::Response) -> Result<T, LangChainError>
where
    T: for<'de> Deserialize<'de>,
{
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(LangChainError::request(format!(
            "hub request failed with status {}: {}",
            status.as_u16(),
            body
        )));
    }

    response.json::<T>().await.map_err(|error| {
        LangChainError::request(format!("failed to decode hub response JSON: {error}"))
    })
}

fn http_error(error: reqwest::Error) -> LangChainError {
    LangChainError::request(format!("hub HTTP request failed: {error}"))
}

impl HubPullResponse {
    fn into_prompt_template(self) -> Result<PromptTemplate, LangChainError> {
        let mut metadata = PromptMetadata::new();
        if let Some(owner) = self.owner {
            metadata.insert("lc_hub_owner".to_owned(), owner);
        }
        if let Some(repo) = self.repo {
            metadata.insert("lc_hub_repo".to_owned(), repo);
        }
        if let Some(commit_hash) = self.commit_hash {
            metadata.insert("lc_hub_commit_hash".to_owned(), commit_hash);
        }
        self.manifest.into_prompt_template(metadata)
    }
}
