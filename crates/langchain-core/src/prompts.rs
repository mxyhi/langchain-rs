use std::collections::HashMap;

use crate::LangChainError;
use crate::messages::{BaseMessage, HumanMessage, SystemMessage};
use crate::runnables::{Runnable, RunnableConfig};

pub type PromptArguments = HashMap<String, PromptArgument>;

#[derive(Debug, Clone, PartialEq)]
pub enum PromptArgument {
    String(String),
    Messages(Vec<BaseMessage>),
}

impl PromptArgument {
    fn as_messages(&self, name: &str) -> Result<&[BaseMessage], LangChainError> {
        match self {
            Self::Messages(value) => Ok(value.as_slice()),
            Self::String(_) => Err(LangChainError::InvalidPromptValue {
                name: name.to_owned(),
                expected: "a message list",
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptTemplate {
    template: String,
}

impl PromptTemplate {
    pub fn new(template: impl Into<String>) -> Self {
        Self {
            template: template.into(),
        }
    }

    pub fn format(&self, arguments: &PromptArguments) -> Result<String, LangChainError> {
        render_template(&self.template, arguments)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MessagesPlaceholder {
    variable_name: String,
}

impl MessagesPlaceholder {
    pub fn new(variable_name: impl Into<String>) -> Self {
        Self {
            variable_name: variable_name.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromptMessageTemplate {
    System(PromptTemplate),
    Human(PromptTemplate),
    MessagesPlaceholder(MessagesPlaceholder),
}

impl PromptMessageTemplate {
    pub fn system(template: impl Into<String>) -> Self {
        Self::System(PromptTemplate::new(template))
    }

    pub fn human(template: impl Into<String>) -> Self {
        Self::Human(PromptTemplate::new(template))
    }

    pub fn placeholder(variable_name: impl Into<String>) -> Self {
        Self::MessagesPlaceholder(MessagesPlaceholder::new(variable_name))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatPromptTemplate {
    messages: Vec<PromptMessageTemplate>,
}

impl ChatPromptTemplate {
    pub fn from_messages(messages: impl IntoIterator<Item = PromptMessageTemplate>) -> Self {
        Self {
            messages: messages.into_iter().collect(),
        }
    }

    pub fn format_messages(
        &self,
        arguments: &PromptArguments,
    ) -> Result<Vec<BaseMessage>, LangChainError> {
        let mut rendered = Vec::new();

        for template in &self.messages {
            match template {
                PromptMessageTemplate::System(template) => {
                    rendered.push(SystemMessage::new(template.format(arguments)?).into());
                }
                PromptMessageTemplate::Human(template) => {
                    rendered.push(HumanMessage::new(template.format(arguments)?).into());
                }
                PromptMessageTemplate::MessagesPlaceholder(placeholder) => {
                    let messages = arguments
                        .get(&placeholder.variable_name)
                        .ok_or_else(|| LangChainError::MissingPromptVariable {
                            name: placeholder.variable_name.clone(),
                        })?
                        .as_messages(&placeholder.variable_name)?;
                    rendered.extend(messages.iter().cloned());
                }
            }
        }

        Ok(rendered)
    }
}

impl Runnable<PromptArguments, Vec<BaseMessage>> for ChatPromptTemplate {
    fn invoke<'a>(
        &'a self,
        input: PromptArguments,
        _config: RunnableConfig,
    ) -> futures_util::future::BoxFuture<'a, Result<Vec<BaseMessage>, LangChainError>> {
        Box::pin(async move { self.format_messages(&input) })
    }
}

fn render_template(template: &str, arguments: &PromptArguments) -> Result<String, LangChainError> {
    let mut rendered = template.to_owned();

    for (name, value) in arguments {
        if let PromptArgument::String(value) = value {
            rendered = rendered.replace(&format!("{{{name}}}"), value);
        }
    }

    if let Some(unresolved) = extract_unresolved_variable(&rendered) {
        return Err(LangChainError::MissingPromptVariable { name: unresolved });
    }

    if rendered.contains('{') || rendered.contains('}') {
        return Err(LangChainError::UnresolvedTemplateVariables { template: rendered });
    }

    Ok(rendered)
}

fn extract_unresolved_variable(template: &str) -> Option<String> {
    let start = template.find('{')?;
    let rest = &template[start + 1..];
    let end = rest.find('}')?;
    Some(rest[..end].to_owned())
}
