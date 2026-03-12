use std::fmt::{Display, Formatter};

use langchain_core::LangChainError;
use langchain_core::messages::{AIMessage, BaseMessage, ToolMessage};
use serde_json::Value;

use super::types::{AgentMiddleware, ModelRequest, ModelResponse};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PIIDetectionError {
    kind: String,
    value: String,
}

impl PIIDetectionError {
    pub fn new(kind: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            value: value.into(),
        }
    }

    pub fn kind(&self) -> &str {
        &self.kind
    }

    pub fn value(&self) -> &str {
        &self.value
    }
}

impl Display for PIIDetectionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "PII detected for rule '{}': '{}'",
            self.kind, self.value
        )
    }
}

impl std::error::Error for PIIDetectionError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RedactionRule {
    label: String,
    pattern: String,
    replacement: String,
    blocking: bool,
}

impl RedactionRule {
    pub fn new(label: impl Into<String>, pattern: impl Into<String>) -> Self {
        let label = label.into();
        Self {
            replacement: format!("[REDACTED_{}]", label.to_ascii_uppercase()),
            label,
            pattern: pattern.into(),
            blocking: false,
        }
    }

    pub fn with_replacement(mut self, replacement: impl Into<String>) -> Self {
        self.replacement = replacement.into();
        self
    }

    pub fn as_blocking(mut self) -> Self {
        self.blocking = true;
        self
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    pub fn replacement(&self) -> &str {
        &self.replacement
    }

    pub fn blocking(&self) -> bool {
        self.blocking
    }

    pub fn redact(&self, text: &str) -> Result<String, PIIDetectionError> {
        let matches = find_matches(text, &self.label, &self.pattern);
        if matches.is_empty() {
            return Ok(text.to_owned());
        }

        if self.blocking {
            let first = &text[matches[0].0..matches[0].1];
            return Err(PIIDetectionError::new(self.label.clone(), first.to_owned()));
        }

        let mut output = String::with_capacity(text.len());
        let mut cursor = 0;
        for (start, end) in matches {
            output.push_str(&text[cursor..start]);
            output.push_str(&self.replacement);
            cursor = end;
        }
        output.push_str(&text[cursor..]);
        Ok(output)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PIIMiddleware {
    rules: Vec<RedactionRule>,
}

impl PIIMiddleware {
    pub fn new(rules: Vec<RedactionRule>) -> Self {
        Self { rules }
    }

    pub fn new_email_redaction() -> Self {
        Self::new(vec![RedactionRule::new(
            "email",
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        )])
    }

    pub fn with_rule(mut self, rule: RedactionRule) -> Self {
        self.rules.push(rule);
        self
    }

    pub fn rules(&self) -> &[RedactionRule] {
        &self.rules
    }

    pub fn redact_text(&self, text: &str) -> Result<String, PIIDetectionError> {
        self.rules
            .iter()
            .try_fold(text.to_owned(), |current, rule| rule.redact(&current))
    }

    pub fn redact_messages(
        &self,
        messages: &[BaseMessage],
    ) -> Result<Vec<BaseMessage>, PIIDetectionError> {
        messages
            .iter()
            .cloned()
            .map(|message| self.redact_message(message))
            .collect()
    }

    fn redact_message(&self, message: BaseMessage) -> Result<BaseMessage, PIIDetectionError> {
        let redacted = self.redact_text(message.content())?;
        Ok(match message {
            BaseMessage::Human(_) => {
                BaseMessage::from(langchain_core::messages::HumanMessage::new(redacted))
            }
            BaseMessage::System(_) => {
                BaseMessage::from(langchain_core::messages::SystemMessage::new(redacted))
            }
            BaseMessage::Ai(ai) => BaseMessage::from(AIMessage::with_parts(
                redacted,
                ai.response_metadata().clone(),
                ai.usage_metadata().cloned(),
                ai.tool_calls().to_vec(),
                ai.invalid_tool_calls().to_vec(),
            )),
            BaseMessage::Tool(tool) => BaseMessage::from(ToolMessage::with_parts(
                redacted,
                tool.tool_call_id().to_owned(),
                tool.name().map(str::to_owned),
                tool.artifact().cloned(),
                tool.status(),
            )),
        })
    }

    pub fn redact_value(&self, value: &Value) -> Result<Value, PIIDetectionError> {
        match value {
            Value::String(text) => self.redact_text(text).map(Value::String),
            Value::Array(items) => items
                .iter()
                .map(|item| self.redact_value(item))
                .collect::<Result<Vec<_>, _>>()
                .map(Value::Array),
            Value::Object(map) => map
                .iter()
                .map(|(key, value)| self.redact_value(value).map(|value| (key.clone(), value)))
                .collect::<Result<serde_json::Map<_, _>, _>>()
                .map(Value::Object),
            _ => Ok(value.clone()),
        }
    }
}

impl AgentMiddleware for PIIMiddleware {
    fn before_model(
        &self,
        request: &mut ModelRequest,
    ) -> Result<Option<super::types::JumpTo>, LangChainError> {
        let redacted = self
            .redact_messages(request.messages())
            .map_err(|error| LangChainError::request(error.to_string()))?;
        *request.messages_mut() = redacted;
        Ok(None)
    }

    fn after_model(
        &self,
        _state: &mut crate::agents::AgentState,
        response: &mut ModelResponse,
    ) -> Result<Option<super::types::JumpTo>, LangChainError> {
        let redacted = self
            .redact_messages(response.result())
            .map_err(|error| LangChainError::request(error.to_string()))?;
        *response.result_mut() = redacted;
        Ok(None)
    }
}

fn find_matches(text: &str, label: &str, pattern: &str) -> Vec<(usize, usize)> {
    if looks_like_email_rule(label, pattern) {
        return email_ranges(text);
    }

    if pattern.is_empty() {
        return Vec::new();
    }

    let mut matches = Vec::new();
    let mut offset = 0;
    while let Some(index) = text[offset..].find(pattern) {
        let start = offset + index;
        let end = start + pattern.len();
        matches.push((start, end));
        offset = end;
    }
    matches
}

fn looks_like_email_rule(label: &str, pattern: &str) -> bool {
    let label = label.to_ascii_lowercase();
    label.contains("email")
        || pattern.contains("@[")
        || pattern.contains('@')
        || pattern.contains("%+-")
}

// We avoid pulling in a regex dependency for middleware helpers. The scanner is
// intentionally small but still catches standard email-like tokens.
fn email_ranges(text: &str) -> Vec<(usize, usize)> {
    let mut matches = Vec::new();
    let mut token_start = None;

    for (index, ch) in text.char_indices() {
        if is_email_token_char(ch) {
            token_start.get_or_insert(index);
            continue;
        }

        if let Some(start) = token_start.take() {
            maybe_push_email_match(text, start, index, &mut matches);
        }
    }

    if let Some(start) = token_start {
        maybe_push_email_match(text, start, text.len(), &mut matches);
    }

    matches
}

fn maybe_push_email_match(text: &str, start: usize, end: usize, matches: &mut Vec<(usize, usize)>) {
    let candidate = &text[start..end];
    if is_email_candidate(candidate) {
        matches.push((start, end));
    }
}

fn is_email_token_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '%' | '+' | '-' | '@')
}

fn is_email_candidate(candidate: &str) -> bool {
    let Some((local, domain)) = candidate.split_once('@') else {
        return false;
    };
    if local.is_empty() || domain.is_empty() || !domain.contains('.') {
        return false;
    }

    local
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '%' | '+' | '-'))
        && domain
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-'))
        && domain.split('.').all(|segment| !segment.is_empty())
}
