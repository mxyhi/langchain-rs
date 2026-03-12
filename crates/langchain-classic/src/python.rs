use std::process::Command;

use langchain_core::LangChainError;

#[derive(Debug, Clone)]
pub struct PythonREPL {
    command: String,
}

impl Default for PythonREPL {
    fn default() -> Self {
        Self::new()
    }
}

impl PythonREPL {
    pub fn new() -> Self {
        Self {
            command: "python3".to_owned(),
        }
    }

    pub fn with_command(mut self, command: impl Into<String>) -> Self {
        self.command = command.into();
        self
    }

    pub async fn run(&self, code: &str) -> Result<String, LangChainError> {
        let command = self.command.clone();
        let code = code.to_owned();

        tokio::task::spawn_blocking(move || {
            let output = Command::new(&command)
                .arg("-c")
                .arg(&code)
                .output()
                .map_err(|error| LangChainError::request(error.to_string()))?;

            if output.status.success() {
                Ok(String::from_utf8_lossy(&output.stdout).to_string())
            } else {
                Err(LangChainError::request(
                    String::from_utf8_lossy(&output.stderr).trim().to_owned(),
                ))
            }
        })
        .await
        .map_err(|error| LangChainError::request(error.to_string()))?
    }
}
