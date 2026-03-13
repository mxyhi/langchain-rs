use std::collections::BTreeMap;
use std::mem;

use langchain_core::LangChainError;
use langchain_core::messages::ToolCall;
use langchain_core::tools::ToolDefinition;
use serde_json::{Map, Value, json};

pub const SYSTEM_PROMPT_FORMAT: &str = "In this environment you have access to a set of tools you can use to answer the user's question.\n\nYou may call them like this:\n<function_calls>\n<invoke>\n<tool_name>$TOOL_NAME</tool_name>\n<parameters>\n<$PARAMETER_NAME>$PARAMETER_VALUE</$PARAMETER_NAME>\n...\n</parameters>\n</invoke>\n</function_calls>\n\nHere are the tools available:\n<tools>\n{formatted_tools}\n</tools>";

pub const TOOL_FORMAT: &str = "<tool_description>\n<tool_name>{tool_name}</tool_name>\n<description>{tool_description}</description>\n<parameters>\n{formatted_parameters}\n</parameters>\n</tool_description>";

pub const TOOL_PARAMETER_FORMAT: &str = "<parameter>\n<name>{parameter_name}</name>\n<type>{parameter_type}</type>\n<description>{parameter_description}</description>\n</parameter>";

pub fn get_system_message(tools: &[ToolDefinition]) -> String {
    let formatted_tools = tools
        .iter()
        .map(|tool| {
            let formatted_parameters = tool
                .parameters()
                .get("properties")
                .and_then(Value::as_object)
                .map(|properties| {
                    properties
                        .iter()
                        .map(|(name, parameter)| {
                            TOOL_PARAMETER_FORMAT
                                .replace("{parameter_name}", name)
                                .replace("{parameter_type}", &parameter_type(parameter))
                                .replace(
                                    "{parameter_description}",
                                    parameter
                                        .get("description")
                                        .and_then(Value::as_str)
                                        .unwrap_or(""),
                                )
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                })
                .unwrap_or_default();

            TOOL_FORMAT
                .replace("{tool_name}", tool.name())
                .replace("{tool_description}", tool.description())
                .replace("{formatted_parameters}", &formatted_parameters)
        })
        .collect::<Vec<_>>()
        .join("\n");

    SYSTEM_PROMPT_FORMAT.replace("{formatted_tools}", &formatted_tools)
}

pub fn extract_tool_calls(
    xml: &str,
    tools: &[ToolDefinition],
) -> Result<Vec<ToolCall>, LangChainError> {
    let nodes = parse_nodes(xml)?;
    let invokes = find_invocations(&nodes);

    invokes
        .into_iter()
        .map(|invoke| {
            let tool_name = invoke
                .child("tool_name")
                .and_then(|node| node.scalar_text())
                .ok_or_else(|| LangChainError::request("missing <tool_name> in <invoke> block"))?;
            let mut args = invoke
                .child("parameters")
                .map(XmlNode::to_json)
                .unwrap_or_else(|| Value::Object(Map::new()));

            if let Some(tool) = tools.iter().find(|tool| tool.name() == tool_name) {
                normalize_value(&mut args, tool.parameters());
            }

            Ok(ToolCall::new(tool_name.to_owned(), args))
        })
        .collect()
}

fn parameter_type(parameter: &Value) -> String {
    if let Some(kind) = parameter.get("type").and_then(Value::as_str) {
        return kind.to_owned();
    }
    if let Some(any_of) = parameter.get("anyOf") {
        return json!({ "anyOf": any_of }).to_string();
    }
    if let Some(all_of) = parameter.get("allOf") {
        return json!({ "allOf": all_of }).to_string();
    }
    parameter.to_string()
}

fn normalize_value(value: &mut Value, schema: &Value) {
    match schema.get("type").and_then(Value::as_str) {
        Some("array") => {
            if !value.is_array() {
                let inner = mem::replace(value, Value::Null);
                *value = Value::Array(vec![inner]);
            }
            if let (Some(items), Value::Array(elements)) = (schema.get("items"), value) {
                for element in elements {
                    normalize_value(element, items);
                }
            }
        }
        Some("object") => {
            if let (Some(properties), Value::Object(object)) =
                (schema.get("properties").and_then(Value::as_object), value)
            {
                for (key, child) in object.iter_mut() {
                    if let Some(child_schema) = properties.get(key) {
                        normalize_value(child, child_schema);
                    }
                }
            }
        }
        _ => {
            if let Value::Object(object) = value {
                if object.len() == 1 {
                    let inner = object.values().next().cloned().unwrap_or(Value::Null);
                    *value = inner;
                }
            }
        }
    }
}

fn find_invocations<'a>(nodes: &'a [XmlNode]) -> Vec<&'a XmlNode> {
    if let Some(function_calls) = nodes.iter().find(|node| node.tag == "function_calls") {
        return function_calls
            .children
            .iter()
            .filter(|node| node.tag == "invoke")
            .collect();
    }

    nodes.iter().filter(|node| node.tag == "invoke").collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct XmlNode {
    tag: String,
    text: String,
    children: Vec<XmlNode>,
}

impl XmlNode {
    fn child(&self, name: &str) -> Option<&Self> {
        self.children.iter().find(|child| child.tag == name)
    }

    fn scalar_text(&self) -> Option<&str> {
        if self.children.is_empty() {
            Some(self.text.trim())
        } else {
            None
        }
    }

    fn to_json(&self) -> Value {
        if self.children.is_empty() {
            return Value::String(self.text.trim().to_owned());
        }

        let mut object = Map::new();
        let mut repeated = BTreeMap::<String, Vec<Value>>::new();

        for child in &self.children {
            repeated
                .entry(child.tag.clone())
                .or_default()
                .push(child.to_json());
        }

        for (tag, values) in repeated {
            if values.len() == 1 {
                object.insert(tag, values.into_iter().next().unwrap_or(Value::Null));
            } else {
                object.insert(tag, Value::Array(values));
            }
        }

        Value::Object(object)
    }
}

fn parse_nodes(xml: &str) -> Result<Vec<XmlNode>, LangChainError> {
    let mut cursor = 0;
    parse_node_list(xml, &mut cursor, None)
}

fn parse_node_list(
    xml: &str,
    cursor: &mut usize,
    closing_tag: Option<&str>,
) -> Result<Vec<XmlNode>, LangChainError> {
    let mut nodes = Vec::new();

    loop {
        skip_whitespace(xml, cursor);
        if *cursor >= xml.len() {
            return match closing_tag {
                Some(tag) => Err(LangChainError::request(format!(
                    "missing closing tag </{tag}> in experimental Anthropic XML"
                ))),
                None => Ok(nodes),
            };
        }

        if xml[*cursor..].starts_with("</") {
            let tag = parse_closing_tag(xml, cursor)?;
            match closing_tag {
                Some(expected) if expected == tag => return Ok(nodes),
                Some(expected) => {
                    return Err(LangChainError::request(format!(
                        "unexpected closing tag </{tag}>; expected </{expected}>"
                    )));
                }
                None => {
                    return Err(LangChainError::request(format!(
                        "unexpected closing tag </{tag}>"
                    )));
                }
            }
        }

        nodes.push(parse_node(xml, cursor)?);
    }
}

fn parse_node(xml: &str, cursor: &mut usize) -> Result<XmlNode, LangChainError> {
    let tag = parse_opening_tag(xml, cursor)?;
    let mut children = Vec::new();
    let mut text = String::new();

    loop {
        if *cursor >= xml.len() {
            return Err(LangChainError::request(format!(
                "missing closing tag </{tag}> in experimental Anthropic XML"
            )));
        }

        if xml[*cursor..].starts_with("</") {
            let closing = parse_closing_tag(xml, cursor)?;
            if closing != tag {
                return Err(LangChainError::request(format!(
                    "unexpected closing tag </{closing}>; expected </{tag}>"
                )));
            }
            return Ok(XmlNode {
                tag,
                text,
                children,
            });
        }

        if xml[*cursor..].starts_with('<') {
            children.push(parse_node(xml, cursor)?);
            continue;
        }

        let next = xml[*cursor..]
            .find('<')
            .map(|offset| *cursor + offset)
            .unwrap_or(xml.len());
        text.push_str(&xml[*cursor..next]);
        *cursor = next;
    }
}

fn parse_opening_tag(xml: &str, cursor: &mut usize) -> Result<String, LangChainError> {
    if !xml[*cursor..].starts_with('<') || xml[*cursor..].starts_with("</") {
        return Err(LangChainError::request(
            "expected opening tag in experimental Anthropic XML",
        ));
    }
    *cursor += 1;
    let end = xml[*cursor..]
        .find('>')
        .map(|offset| *cursor + offset)
        .ok_or_else(|| LangChainError::request("unterminated opening tag"))?;
    let tag = xml[*cursor..end].trim();
    if tag.is_empty() || tag.contains(' ') {
        return Err(LangChainError::request(format!(
            "unsupported tag `{tag}` in experimental Anthropic XML"
        )));
    }
    *cursor = end + 1;
    Ok(tag.to_owned())
}

fn parse_closing_tag(xml: &str, cursor: &mut usize) -> Result<String, LangChainError> {
    if !xml[*cursor..].starts_with("</") {
        return Err(LangChainError::request(
            "expected closing tag in experimental Anthropic XML",
        ));
    }
    *cursor += 2;
    let end = xml[*cursor..]
        .find('>')
        .map(|offset| *cursor + offset)
        .ok_or_else(|| LangChainError::request("unterminated closing tag"))?;
    let tag = xml[*cursor..end].trim();
    *cursor = end + 1;
    Ok(tag.to_owned())
}

fn skip_whitespace(xml: &str, cursor: &mut usize) {
    while let Some(character) = xml[*cursor..].chars().next() {
        if !character.is_whitespace() {
            break;
        }
        *cursor += character.len_utf8();
    }
}
