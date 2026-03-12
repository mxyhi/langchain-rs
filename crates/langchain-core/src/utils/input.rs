use std::collections::BTreeMap;
use std::io::Write;

const TEXT_COLOR_MAPPING: &[(&str, &str)] = &[
    ("blue", "36;1"),
    ("yellow", "33;1"),
    ("pink", "38;5;200"),
    ("green", "32;1"),
    ("red", "31;1"),
];

pub fn get_color_mapping(
    items: Vec<String>,
    excluded_colors: Option<Vec<String>>,
) -> Result<BTreeMap<String, String>, String> {
    let mut colors = TEXT_COLOR_MAPPING
        .iter()
        .map(|(name, _)| (*name).to_owned())
        .collect::<Vec<_>>();

    if let Some(excluded_colors) = excluded_colors {
        colors.retain(|color| !excluded_colors.contains(color));
    }

    if colors.is_empty() {
        return Err("No colors available after applying exclusions.".to_owned());
    }

    Ok(items
        .into_iter()
        .enumerate()
        .map(|(index, item)| (item, colors[index % colors.len()].clone()))
        .collect())
}

pub fn get_colored_text(text: &str, color: &str) -> String {
    let color_code = TEXT_COLOR_MAPPING
        .iter()
        .find(|(name, _)| *name == color)
        .map(|(_, code)| *code)
        .unwrap_or("0");
    format!("\u{001b}[{color_code}m\u{001b}[1;3m{text}\u{001b}[0m")
}

pub fn get_bolded_text(text: &str) -> String {
    format!("\u{001b}[1m{text}\u{001b}[0m")
}

pub fn print_text(text: &str, color: Option<&str>, end: &str, file: Option<&mut dyn Write>) {
    let rendered = match color {
        Some(color) => get_colored_text(text, color),
        None => text.to_owned(),
    };

    match file {
        Some(file) => {
            let _ = write!(file, "{rendered}{end}");
            let _ = file.flush();
        }
        None => {
            print!("{rendered}{end}");
        }
    }
}
