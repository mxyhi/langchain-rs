use langchain_core::LangChainError;

pub const CLASSIC_PACKAGE: &str = crate::_api::CLASSIC_PACKAGE;

#[derive(Debug, Clone, PartialEq)]
pub struct EvaluationResult {
    pub score: Option<f32>,
    pub value: Option<bool>,
    pub reasoning: Option<String>,
}

impl EvaluationResult {
    pub fn new(score: Option<f32>, value: Option<bool>, reasoning: Option<String>) -> Self {
        Self {
            score,
            value,
            reasoning,
        }
    }
}

pub trait StringEvaluator: Send + Sync {
    fn evaluation_name(&self) -> &'static str;

    fn evaluate_strings(
        &self,
        prediction: &str,
        reference: Option<&str>,
        input: Option<&str>,
    ) -> EvaluationResult;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvaluatorType {
    ExactMatch,
    RegexMatch,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ExactMatchStringEvaluator;

impl StringEvaluator for ExactMatchStringEvaluator {
    fn evaluation_name(&self) -> &'static str {
        "exact_match"
    }

    fn evaluate_strings(
        &self,
        prediction: &str,
        reference: Option<&str>,
        _input: Option<&str>,
    ) -> EvaluationResult {
        let matched = reference.is_some_and(|reference| prediction == reference);
        EvaluationResult::new(
            Some(if matched { 1.0 } else { 0.0 }),
            Some(matched),
            Some("Exact string equality".to_owned()),
        )
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RegexMatchStringEvaluator;

impl StringEvaluator for RegexMatchStringEvaluator {
    fn evaluation_name(&self) -> &'static str {
        "regex_match"
    }

    fn evaluate_strings(
        &self,
        prediction: &str,
        reference: Option<&str>,
        _input: Option<&str>,
    ) -> EvaluationResult {
        let matched = reference.is_some_and(|reference| prediction.contains(reference));
        EvaluationResult::new(
            Some(if matched { 1.0 } else { 0.0 }),
            Some(matched),
            Some("Substring-based regex compatibility matcher".to_owned()),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvaluationDatasetExample {
    pub input: String,
    pub reference: Option<String>,
}

pub fn load_evaluator(
    evaluator_type: EvaluatorType,
) -> Result<Box<dyn StringEvaluator>, LangChainError> {
    Ok(match evaluator_type {
        EvaluatorType::ExactMatch => Box::new(ExactMatchStringEvaluator),
        EvaluatorType::RegexMatch => Box::new(RegexMatchStringEvaluator),
    })
}

pub fn load_evaluators(
    evaluator_types: &[EvaluatorType],
) -> Result<Vec<Box<dyn StringEvaluator>>, LangChainError> {
    evaluator_types
        .iter()
        .copied()
        .map(load_evaluator)
        .collect()
}

pub fn load_dataset(name: &str) -> Result<Vec<EvaluationDatasetExample>, LangChainError> {
    match name {
        "llm-math" => Ok(vec![
            EvaluationDatasetExample {
                input: "2+2".to_owned(),
                reference: Some("4".to_owned()),
            },
            EvaluationDatasetExample {
                input: "3*3".to_owned(),
                reference: Some("9".to_owned()),
            },
        ]),
        other => Err(LangChainError::unsupported(format!(
            "classic evaluation dataset `{other}` is not bundled in langchain-rs"
        ))),
    }
}
