use std::process::Command;
use std::sync::LazyLock;

fn parse_major_version(output: &[u8]) -> Option<usize> {
    String::from_utf8_lossy(output).trim().parse().ok()
}

fn probe_interpreter(interpreter: &str) -> Option<usize> {
    let output = Command::new(interpreter)
        .args([
            "-c",
            "import pydantic; print(int(pydantic.__version__.split('.')[0]))",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    parse_major_version(&output.stdout)
}

pub fn get_pydantic_major_version() -> usize {
    ["python3", "python"]
        .into_iter()
        .find_map(probe_interpreter)
        .unwrap_or(0)
}

pub static PYDANTIC_MAJOR_VERSION: LazyLock<usize> = LazyLock::new(get_pydantic_major_version);
