use std::collections::BTreeMap;

use crate::LangChainError;

#[derive(Debug, Clone, Copy, Default)]
pub struct StrictFormatter;

impl StrictFormatter {
    pub fn new() -> Self {
        Self
    }

    pub fn format(
        &self,
        template: &str,
        arguments: &BTreeMap<String, String>,
    ) -> Result<String, LangChainError> {
        let mut rendered = template.to_owned();
        for (name, value) in arguments {
            rendered = rendered.replace(&format!("{{{name}}}"), value);
        }

        if let Some(unresolved) = extract_unresolved_variable(&rendered) {
            return Err(LangChainError::MissingPromptVariable { name: unresolved });
        }

        Ok(rendered)
    }

    pub fn validate_input_variables(
        &self,
        template: &str,
        input_variables: &[String],
    ) -> Result<(), LangChainError> {
        let arguments = input_variables
            .iter()
            .map(|name| (name.clone(), "foo".to_owned()))
            .collect::<BTreeMap<_, _>>();
        self.format(template, &arguments).map(|_| ())
    }
}

pub fn formatter() -> StrictFormatter {
    StrictFormatter::new()
}

fn extract_unresolved_variable(template: &str) -> Option<String> {
    let start = template.find('{')?;
    let rest = &template[start + 1..];
    let end = rest.find('}')?;
    Some(rest[..end].to_owned())
}
