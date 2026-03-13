#[test]
fn facade_exposes_package_version_like_python_dunder_version() {
    // Match `langchain.__version__`: the facade should expose the package version.
    assert_eq!(langchain::VERSION, env!("CARGO_PKG_VERSION"));
}
