use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use langchain_core::LangChainError;

use super::types::AgentMiddleware;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShellCommandOutput {
    status_code: i32,
    stdout: String,
    stderr: String,
}

impl ShellCommandOutput {
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

    pub fn success(&self) -> bool {
        self.status_code == 0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodexSandboxExecutionPolicy {
    shell: String,
    working_directory: Option<PathBuf>,
    environment: BTreeMap<String, String>,
    allow_network: bool,
    allow_writes: bool,
    block_dangerous_commands: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DockerExecutionPolicy;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct HostExecutionPolicy;

impl Default for CodexSandboxExecutionPolicy {
    fn default() -> Self {
        Self {
            shell: "/bin/zsh".to_owned(),
            working_directory: None,
            environment: BTreeMap::new(),
            allow_network: false,
            allow_writes: false,
            block_dangerous_commands: true,
        }
    }
}

impl CodexSandboxExecutionPolicy {
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

    pub fn allow_network(mut self) -> Self {
        self.allow_network = true;
        self
    }

    pub fn allow_writes(mut self) -> Self {
        self.allow_writes = true;
        self
    }

    pub fn shell(&self) -> &str {
        &self.shell
    }

    pub fn working_directory(&self) -> Option<&Path> {
        self.working_directory.as_deref()
    }

    pub fn validate(&self, command_line: &str) -> Result<(), LangChainError> {
        let trimmed = command_line.trim();
        if trimmed.is_empty() {
            return Err(LangChainError::request("shell command cannot be empty"));
        }

        // The sandbox policy is heuristic by design. We block obviously destructive
        // mutations here and leave precise allow-listing to higher-level orchestration.
        if self.block_dangerous_commands {
            let lowered = trimmed.to_ascii_lowercase();
            if ["rm -rf /", "shutdown", "reboot", "mkfs", ":(){", "dd if="]
                .iter()
                .any(|needle| lowered.contains(needle))
            {
                return Err(LangChainError::unsupported(
                    "command rejected by sandbox dangerous-command policy",
                ));
            }
        }

        if !self.allow_writes && looks_mutating(trimmed) {
            return Err(LangChainError::unsupported(
                "command rejected because write access is disabled",
            ));
        }

        if !self.allow_network && looks_networked(trimmed) {
            return Err(LangChainError::unsupported(
                "command rejected because network access is disabled",
            ));
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShellToolMiddleware {
    policy: CodexSandboxExecutionPolicy,
}

impl ShellToolMiddleware {
    pub fn new(policy: CodexSandboxExecutionPolicy) -> Self {
        Self { policy }
    }

    pub fn policy(&self) -> &CodexSandboxExecutionPolicy {
        &self.policy
    }

    pub fn execute(&self, command_line: &str) -> Result<ShellCommandOutput, LangChainError> {
        self.policy.validate(command_line)?;

        let mut command = Command::new(self.policy.shell());
        command.arg("-lc").arg(command_line);
        if let Some(working_directory) = self.policy.working_directory() {
            command.current_dir(working_directory);
        }
        command.envs(self.policy.environment.clone());

        let output = command.output().map_err(|error| {
            LangChainError::request(format!("failed to execute shell command: {error}"))
        })?;

        Ok(ShellCommandOutput::new(
            output.status.code().unwrap_or(1),
            String::from_utf8_lossy(&output.stdout).into_owned(),
            String::from_utf8_lossy(&output.stderr).into_owned(),
        ))
    }
}

impl AgentMiddleware for ShellToolMiddleware {}

fn looks_mutating(command_line: &str) -> bool {
    let lowered = command_line.to_ascii_lowercase();
    [
        "rm ",
        "mv ",
        "cp ",
        "touch ",
        "mkdir ",
        "rmdir ",
        "tee ",
        "sed -i",
        "perl -pi",
        "truncate ",
        "git commit",
        ">",
    ]
    .iter()
    .any(|needle| lowered.contains(needle))
}

fn looks_networked(command_line: &str) -> bool {
    let lowered = command_line.to_ascii_lowercase();
    ["curl ", "wget ", "nc ", "ssh ", "scp ", "rsync ", "ping "]
        .iter()
        .any(|needle| lowered.contains(needle))
}
