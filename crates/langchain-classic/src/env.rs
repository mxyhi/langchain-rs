#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeEnvironment {
    pub library_version: String,
    pub library: String,
    pub platform: String,
    pub runtime: String,
    pub runtime_version: String,
}

pub fn get_runtime_environment() -> RuntimeEnvironment {
    RuntimeEnvironment {
        library_version: crate::VERSION.to_owned(),
        library: "langchain-classic".to_owned(),
        platform: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        runtime: "rust".to_owned(),
        runtime_version: std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "unknown".to_owned()),
    }
}
