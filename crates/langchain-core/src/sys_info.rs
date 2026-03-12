pub fn system_info_report(additional_packages: &[&str]) -> String {
    let mut lines = vec![
        "System Information".to_owned(),
        "------------------".to_owned(),
        format!("> OS: {}", std::env::consts::OS),
        format!("> Architecture: {}", std::env::consts::ARCH),
        format!(
            "> Runtime: rust ({})",
            std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "unknown".to_owned())
        ),
        "".to_owned(),
        "Package Information".to_owned(),
        "-------------------".to_owned(),
        format!("> langchain-core: {}", crate::VERSION),
    ];

    for package in additional_packages {
        lines.push(format!("> {package}: requested"));
    }

    lines.join("\n")
}

pub fn print_sys_info(additional_packages: &[&str]) {
    println!("{}", system_info_report(additional_packages));
}
