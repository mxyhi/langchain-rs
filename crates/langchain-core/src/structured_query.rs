use serde::{Deserialize, Serialize};
use serde_json::Value;

pub trait Visitor<T> {
    fn visit_operation(&mut self, operation: &Operation) -> T;

    fn visit_comparison(&mut self, comparison: &Comparison) -> T;

    fn visit_structured_query(&mut self, structured_query: &StructuredQuery) -> T;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Operator {
    And,
    Or,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Comparator {
    Eq,
    Ne,
    Gt,
    Gte,
    Lt,
    Lte,
    Contain,
    Like,
    In,
    Nin,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Comparison {
    comparator: Comparator,
    attribute: String,
    value: Value,
}

impl Comparison {
    pub fn new(comparator: Comparator, attribute: impl Into<String>, value: Value) -> Self {
        Self {
            comparator,
            attribute: attribute.into(),
            value,
        }
    }

    pub fn comparator(&self) -> Comparator {
        self.comparator
    }

    pub fn attribute(&self) -> &str {
        &self.attribute
    }

    pub fn value(&self) -> &Value {
        &self.value
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Operation {
    operator: Operator,
    arguments: Vec<FilterDirective>,
}

impl Operation {
    pub fn new(operator: Operator, arguments: Vec<FilterDirective>) -> Self {
        Self {
            operator,
            arguments,
        }
    }

    pub fn operator(&self) -> Operator {
        self.operator
    }

    pub fn arguments(&self) -> &[FilterDirective] {
        &self.arguments
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FilterDirective {
    Comparison(Comparison),
    Operation(Operation),
}

impl FilterDirective {
    pub fn accept<T>(&self, visitor: &mut impl Visitor<T>) -> T {
        match self {
            Self::Comparison(comparison) => visitor.visit_comparison(comparison),
            Self::Operation(operation) => visitor.visit_operation(operation),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuredQuery {
    query: String,
    filter: Option<FilterDirective>,
    limit: Option<usize>,
}

impl StructuredQuery {
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            filter: None,
            limit: None,
        }
    }

    pub fn with_filter(mut self, filter: FilterDirective) -> Self {
        self.filter = Some(filter);
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    pub fn query(&self) -> &str {
        &self.query
    }

    pub fn filter(&self) -> Option<&FilterDirective> {
        self.filter.as_ref()
    }

    pub fn limit(&self) -> Option<usize> {
        self.limit
    }

    pub fn accept<T>(&self, visitor: &mut impl Visitor<T>) -> T {
        visitor.visit_structured_query(self)
    }
}
