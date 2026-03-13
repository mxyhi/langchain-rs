use crate::evaluation::{EvaluatorType, load_evaluator};
use langchain_core::LangChainError;

pub const CLASSIC_PACKAGE: &str = crate::_api::CLASSIC_PACKAGE;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetExample {
    input: String,
    reference: Option<String>,
}

impl DatasetExample {
    pub fn new(input: impl Into<String>, reference: Option<impl Into<String>>) -> Self {
        Self {
            input: input.into(),
            reference: reference.map(Into::into),
        }
    }

    pub fn input(&self) -> &str {
        &self.input
    }

    pub fn reference(&self) -> Option<&str> {
        self.reference.as_deref()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RunEvalConfig {
    evaluators: Vec<EvaluatorType>,
}

impl RunEvalConfig {
    pub fn with_evaluator(mut self, evaluator: EvaluatorType) -> Self {
        self.evaluators.push(evaluator);
        self
    }

    fn primary_evaluator(&self) -> EvaluatorType {
        self.evaluators
            .first()
            .copied()
            .unwrap_or(EvaluatorType::ExactMatch)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DatasetRun {
    pub example: DatasetExample,
    pub score: Option<f32>,
    pub succeeded: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RunSummary {
    pub total_examples: usize,
    pub successful_evaluations: usize,
    pub average_score: Option<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DatasetRunReport {
    runs: Vec<DatasetRun>,
    summary: RunSummary,
}

impl DatasetRunReport {
    pub fn runs(&self) -> &[DatasetRun] {
        &self.runs
    }

    pub fn summary(&self) -> &RunSummary {
        &self.summary
    }
}

pub fn run_on_dataset(
    dataset: &[DatasetExample],
    config: &RunEvalConfig,
) -> Result<DatasetRunReport, LangChainError> {
    let evaluator = load_evaluator(config.primary_evaluator())?;
    let runs = dataset
        .iter()
        .cloned()
        .map(|example| {
            let result = evaluator.evaluate_strings(
                example.input(),
                example.reference(),
                Some(example.input()),
            );
            DatasetRun {
                example,
                score: result.score,
                succeeded: result.score.is_some(),
            }
        })
        .collect::<Vec<_>>();

    let successful_scores = runs.iter().filter_map(|run| run.score).collect::<Vec<_>>();
    let summary = RunSummary {
        total_examples: runs.len(),
        successful_evaluations: runs.iter().filter(|run| run.succeeded).count(),
        average_score: if successful_scores.is_empty() {
            None
        } else {
            Some(successful_scores.iter().sum::<f32>() / successful_scores.len() as f32)
        },
    };

    Ok(DatasetRunReport { runs, summary })
}

pub async fn arun_on_dataset(
    dataset: &[DatasetExample],
    config: &RunEvalConfig,
) -> Result<DatasetRunReport, LangChainError> {
    run_on_dataset(dataset, config)
}
