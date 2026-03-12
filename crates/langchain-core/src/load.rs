use std::any::Any;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::LangChainError;
use crate::documents::Document;
use crate::messages::{AIMessage, ChatMessage, FunctionMessage, HumanMessage, SystemMessage};

pub const LC_SERIALIZATION_VERSION: u8 = 1;

pub type SerializedArguments = Map<String, Value>;
type ConstructorFn = fn(SerializedArguments) -> Result<Box<dyn Serializable>, LangChainError>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SerializedValue {
    Constructor {
        lc: u8,
        id: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        kwargs: SerializedArguments,
    },
    NotImplemented {
        lc: u8,
        id: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        repr: Option<String>,
    },
}

impl SerializedValue {
    pub fn constructor(id: Vec<String>, kwargs: SerializedArguments) -> Self {
        Self::Constructor {
            lc: LC_SERIALIZATION_VERSION,
            id,
            name: None,
            kwargs,
        }
    }

    pub fn not_implemented(id: Vec<String>, repr: Option<String>) -> Self {
        Self::NotImplemented {
            lc: LC_SERIALIZATION_VERSION,
            id,
            repr,
        }
    }

    pub fn id(&self) -> &[String] {
        match self {
            Self::Constructor { id, .. } | Self::NotImplemented { id, .. } => id.as_slice(),
        }
    }
}

pub trait Serializable: fmt::Debug + Send + Sync + 'static {
    fn lc_id(&self) -> Vec<String>;

    fn lc_name(&self) -> Option<String> {
        None
    }

    fn is_lc_serializable(&self) -> bool {
        true
    }

    fn to_serialized_kwargs(&self) -> Result<SerializedArguments, LangChainError>;

    fn to_serialized_value(&self) -> Result<SerializedValue, LangChainError> {
        if self.is_lc_serializable() {
            return Ok(SerializedValue::Constructor {
                lc: LC_SERIALIZATION_VERSION,
                id: self.lc_id(),
                name: self.lc_name(),
                kwargs: self.to_serialized_kwargs()?,
            });
        }

        Ok(SerializedValue::not_implemented(
            self.lc_id(),
            Some(format!("{self:?}")),
        ))
    }

    fn as_any(&self) -> &dyn Any;
}

pub trait Revivable: Serializable + Sized {
    fn static_lc_id() -> Vec<String>;

    fn from_serialized_kwargs(kwargs: SerializedArguments) -> Result<Self, LangChainError>;
}

#[derive(Clone, Default)]
pub struct Reviver {
    constructors: BTreeMap<Vec<String>, ConstructorFn>,
    aliases: BTreeMap<Vec<String>, Vec<String>>,
    allowed_ids: BTreeSet<Vec<String>>,
}

impl fmt::Debug for Reviver {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Reviver")
            .field(
                "constructors",
                &self.constructors.keys().collect::<Vec<_>>(),
            )
            .field("aliases", &self.aliases)
            .field("allowed_ids", &self.allowed_ids)
            .finish()
    }
}

impl Reviver {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn core() -> Self {
        let mut reviver = Self::new();
        reviver
            .register::<Document>()
            .register::<AIMessage>()
            .register::<HumanMessage>()
            .register::<SystemMessage>()
            .register::<ChatMessage>()
            .register::<FunctionMessage>()
            .allow_alias(
                ["langchain", "schema", "document", "Document"],
                Document::static_lc_id(),
            )
            .allow_alias(
                ["langchain", "schema", "messages", "AIMessage"],
                AIMessage::static_lc_id(),
            )
            .allow_alias(
                ["langchain", "schema", "messages", "HumanMessage"],
                HumanMessage::static_lc_id(),
            )
            .allow_alias(
                ["langchain", "schema", "messages", "SystemMessage"],
                SystemMessage::static_lc_id(),
            )
            .allow_alias(
                ["langchain", "schema", "messages", "ChatMessage"],
                ChatMessage::static_lc_id(),
            )
            .allow_alias(
                ["langchain", "schema", "messages", "FunctionMessage"],
                FunctionMessage::static_lc_id(),
            );
        reviver
    }

    pub fn register<T: Revivable>(&mut self) -> &mut Self {
        let id = T::static_lc_id();
        self.allowed_ids.insert(id.clone());
        self.constructors.insert(id, |kwargs| {
            Ok(Box::new(T::from_serialized_kwargs(kwargs)?))
        });
        self
    }

    pub fn allow_alias(
        &mut self,
        legacy_id: impl IntoIterator<Item = impl Into<String>>,
        current_id: Vec<String>,
    ) -> &mut Self {
        self.aliases
            .insert(legacy_id.into_iter().map(Into::into).collect(), current_id);
        self
    }

    pub fn revive(
        &self,
        serialized: SerializedValue,
    ) -> Result<Box<dyn Serializable>, LangChainError> {
        let (id, kwargs) = match serialized {
            SerializedValue::Constructor { id, kwargs, .. } => (self.resolve_id(id)?, kwargs),
            SerializedValue::NotImplemented { id, .. } => {
                return Err(LangChainError::unsupported(format!(
                    "serialized id `{}` is marked as not_implemented",
                    id.join("."),
                )));
            }
        };

        let constructor = self.constructors.get(&id).ok_or_else(|| {
            LangChainError::unsupported(format!(
                "serialized id `{}` is not registered with this reviver",
                id.join("."),
            ))
        })?;
        constructor(kwargs)
    }

    fn resolve_id(&self, id: Vec<String>) -> Result<Vec<String>, LangChainError> {
        let resolved = self.aliases.get(&id).cloned().unwrap_or(id);
        if self.allowed_ids.contains(&resolved) {
            return Ok(resolved);
        }

        Err(LangChainError::unsupported(format!(
            "serialized id `{}` is not allowed by this reviver",
            resolved.join("."),
        )))
    }
}

pub fn dumpd(value: &dyn Serializable) -> Result<SerializedValue, LangChainError> {
    value.to_serialized_value()
}

pub fn dumps(value: &dyn Serializable, pretty: bool) -> Result<String, LangChainError> {
    let serialized = dumpd(value)?;
    if pretty {
        return serde_json::to_string_pretty(&serialized).map_err(Into::into);
    }

    serde_json::to_string(&serialized).map_err(Into::into)
}

pub fn load(
    serialized: SerializedValue,
    reviver: &Reviver,
) -> Result<Box<dyn Serializable>, LangChainError> {
    reviver.revive(serialized)
}

pub fn loads(text: &str, reviver: &Reviver) -> Result<Box<dyn Serializable>, LangChainError> {
    let serialized: SerializedValue = serde_json::from_str(text)?;
    load(serialized, reviver)
}

fn serialize_kwargs<T: Serialize>(value: &T) -> Result<SerializedArguments, LangChainError> {
    match serde_json::to_value(value)? {
        Value::Object(arguments) => Ok(arguments),
        other => Err(LangChainError::request(format!(
            "serializable kwargs must be a JSON object, got {other}",
        ))),
    }
}

fn deserialize_kwargs<T: DeserializeOwned>(
    kwargs: SerializedArguments,
) -> Result<T, LangChainError> {
    serde_json::from_value(Value::Object(kwargs)).map_err(Into::into)
}

macro_rules! impl_serde_serializable {
    ($ty:ty, [$($segment:expr),+ $(,)?]) => {
        impl Serializable for $ty {
            fn lc_id(&self) -> Vec<String> {
                Self::static_lc_id()
            }

            fn to_serialized_kwargs(&self) -> Result<SerializedArguments, LangChainError> {
                serialize_kwargs(self)
            }

            fn as_any(&self) -> &dyn Any {
                self
            }
        }

        impl Revivable for $ty {
            fn static_lc_id() -> Vec<String> {
                vec![$($segment.to_owned()),+]
            }

            fn from_serialized_kwargs(
                kwargs: SerializedArguments,
            ) -> Result<Self, LangChainError> {
                deserialize_kwargs(kwargs)
            }
        }
    };
}

impl_serde_serializable!(
    Document,
    ["langchain_core", "documents", "base", "Document"]
);
impl_serde_serializable!(AIMessage, ["langchain_core", "messages", "ai", "AIMessage"]);
impl_serde_serializable!(
    HumanMessage,
    ["langchain_core", "messages", "human", "HumanMessage"]
);
impl_serde_serializable!(
    SystemMessage,
    ["langchain_core", "messages", "system", "SystemMessage"]
);
impl_serde_serializable!(
    ChatMessage,
    ["langchain_core", "messages", "chat", "ChatMessage"]
);
impl_serde_serializable!(
    FunctionMessage,
    ["langchain_core", "messages", "function", "FunctionMessage"]
);
