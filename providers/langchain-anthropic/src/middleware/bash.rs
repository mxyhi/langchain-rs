use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use langchain_core::LangChainError;

use super::anthropic_tools::AnthropicBuiltInTool;

pub const BASH_TOOL_NAME: &str = "bash";
pub const BASH_TOOL_TYPE: &str = "bash_20250124";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BashExecutionPolicy {
    shell: String,
    working_directory: Option<PathBuf>,
    environment: BTreeMap<String, String>,
    allow_writes: bool,
    allow_network: bool,
}

impl Default for BashExecutionPolicy {
    fn default() -> Self {
        Self {
            shell: "/bin/bash".to_owned(),
            working_directory: None,
            environment: BTreeMap::new(),
            allow_writes: true,
            allow_network: true,
        }
    }
}

impl BashExecutionPolicy {
    pub fn with_shell(mut self, shell: impl Into<String>) -> Self {
        self.shell = shell.into();
        self
    }

    pub fn with_working_directory(mut self, working_directory: impl Into<PathBuf>) -> Self {
        self.working_directory = Some(working_directory.into());
        self
    }

    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.environment.insert(key.into(), value.into());
        self
    }

    pub fn allow_writes(mut self, allow_writes: bool) -> Self {
        self.allow_writes = allow_writes;
        self
    }

    pub fn allow_network(mut self, allow_network: bool) -> Self {
        self.allow_network = allow_network;
        self
    }

    pub fn shell(&self) -> &str {
        &self.shell
    }

    pub fn working_directory(&self) -> Option<&Path> {
        self.working_directory.as_deref()
    }

    fn validate(&self, command_line: &str) -> Result<(), LangChainError> {
        let trimmed = command_line.trim();
        if trimmed.is_empty() {
            return Err(LangChainError::request("bash command cannot be empty"));
        }

        let lowered = trimmed.to_ascii_lowercase();
        if ["rm -rf /", "shutdown", "reboot", "mkfs", ":(){", "dd if="]
            .iter()
            .any(|needle| lowered.contains(needle))
        {
            return Err(LangChainError::unsupported(
                "command rejected by Anthropic bash safety policy",
            ));
        }

        if !self.allow_writes
            && [">", ">>", "mv ", "cp ", "rm ", "touch ", "mkdir "]
                .iter()
                .any(|needle| lowered.contains(needle))
        {
            return Err(LangChainError::unsupported(
                "command rejected because write access is disabled",
            ));
        }

        if !self.allow_network
            && ["curl ", "wget ", "nc ", "ssh ", "scp "]
                .iter()
                .any(|needle| lowered.contains(needle))
        {
            return Err(LangChainError::unsupported(
                "command rejected because network access is disabled",
            ));
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BashToolOutput {
    status_code: i32,
    stdout: String,
    stderr: String,
}

impl BashToolOutput {
    pub fn new(status_code: i32, stdout: impl Into<String>, stderr: impl Into<String>) -> Self {
        Self {
            status_code,
            stdout: stdout.into(),
            stderr: stderr.into(),
        }
    }

    pub fn status_code(&self) -> i32 {
        self.status_code
    }

    pub fn stdout(&self) -> &str {
        &self.stdout
    }

    pub fn stderr(&self) -> &str {
        &self.stderr
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaudeBashToolMiddleware {
    policy: BashExecutionPolicy,
}

impl ClaudeBashToolMiddleware {
    pub fn new<P>(workspace_root: Option<P>) -> Self
    where
        P: AsRef<Path>,
    {
        let policy = workspace_root
            .map(|path| BashExecutionPolicy::default().with_working_directory(path.as_ref()))
            .unwrap_or_default();
        Self { policy }
    }

    pub fn with_policy(mut self, policy: BashExecutionPolicy) -> Self {
        self.policy = policy;
        self
    }

    pub fn policy(&self) -> &BashExecutionPolicy {
        &self.policy
    }

    pub fn tool(&self) -> AnthropicBuiltInTool {
        AnthropicBuiltInTool::new(BASH_TOOL_TYPE, BASH_TOOL_NAME)
    }

    pub fn execute(&self, command_line: &str) -> Result<BashToolOutput, LangChainError> {
        self.policy.validate(command_line)?;

        let mut command = Command::new(self.policy.shell());
        command.arg("-lc").arg(command_line);
        if let Some(working_directory) = self.policy.working_directory() {
            command.current_dir(working_directory);
        }
        command.envs(self.policy.environment.clone());

        let output = command.output().map_err(|error| {
            LangChainError::request(format!("failed to execute Anthropic bash command: {error}"))
        })?;

        Ok(BashToolOutput::new(
            output.status.code().unwrap_or(1),
            String::from_utf8_lossy(&output.stdout).into_owned(),
            String::from_utf8_lossy(&output.stderr).into_owned(),
        ))
    }
}
