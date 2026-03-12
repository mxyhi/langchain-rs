use std::sync::OnceLock;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeEnvironment {
    pub library_version: String,
    pub library: String,
    pub platform: String,
    pub runtime: String,
    pub runtime_version: String,
}

pub const LIBRARY_NAME: &str = "langchain-core";
pub const RUNTIME_NAME: &str = "rust";

pub fn get_runtime_environment() -> RuntimeEnvironment {
    static ENV: OnceLock<RuntimeEnvironment> = OnceLock::new();

    ENV.get_or_init(|| RuntimeEnvironment {
        library_version: crate::VERSION.to_owned(),
        library: LIBRARY_NAME.to_owned(),
        platform: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        runtime: RUNTIME_NAME.to_owned(),
        runtime_version: std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "unknown".to_owned()),
    })
    .clone()
}
