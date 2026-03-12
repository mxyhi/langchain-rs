use langchain_core::LangChainError;
use langchain_core::language_models::BaseLLM;
use langchain_core::prompts::{PromptArgument, PromptArguments, PromptTemplate};
use langchain_core::runnables::RunnableConfig;

use crate::input::get_color_mapping;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelLaboratoryResult {
    name: String,
    output: String,
    color: String,
}

impl ModelLaboratoryResult {
    pub fn new(
        name: impl Into<String>,
        output: impl Into<String>,
        color: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            output: output.into(),
            color: color.into(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn output(&self) -> &str {
        &self.output
    }

    pub fn color(&self) -> &str {
        &self.color
    }
}

pub struct ModelLaboratory {
    llms: Vec<Box<dyn BaseLLM>>,
    names: Vec<String>,
    prompt: PromptTemplate,
}

impl ModelLaboratory {
    pub fn new(
        llms: Vec<Box<dyn BaseLLM>>,
        names: Option<Vec<String>>,
        prompt: Option<PromptTemplate>,
    ) -> Result<Self, LangChainError> {
        if llms.is_empty() {
            return Err(LangChainError::request(
                "model laboratory requires at least one llm",
            ));
        }

        let names = match names {
            Some(names) => {
                if names.len() != llms.len() {
                    return Err(LangChainError::request(
                        "model laboratory names length must match llms length",
                    ));
                }
                names
            }
            None => llms.iter().map(|llm| llm.model_name().to_owned()).collect(),
        };

        Ok(Self {
            llms,
            names,
            prompt: prompt.unwrap_or_else(|| PromptTemplate::new("{_input}")),
        })
    }

    pub fn from_llms(
        llms: Vec<Box<dyn BaseLLM>>,
        prompt: Option<PromptTemplate>,
    ) -> Result<Self, LangChainError> {
        Self::new(llms, None, prompt)
    }

    pub async fn compare(
        &self,
        text: impl Into<String>,
    ) -> Result<Vec<ModelLaboratoryResult>, LangChainError> {
        let text = text.into();
        let color_mapping = get_color_mapping(
            (0..self.llms.len())
                .map(|index| index.to_string())
                .collect(),
            None,
        )
        .map_err(LangChainError::request)?;

        let mut results = Vec::with_capacity(self.llms.len());

        for (index, llm) in self.llms.iter().enumerate() {
            let arguments = PromptArguments::from([(
                "_input".to_owned(),
                PromptArgument::String(text.clone()),
            )]);
            let prompt = self.prompt.format(&arguments)?;
            let output = llm.invoke_prompt(prompt, RunnableConfig::default()).await?;

            results.push(ModelLaboratoryResult::new(
                self.names[index].clone(),
                output,
                color_mapping
                    .get(&index.to_string())
                    .cloned()
                    .unwrap_or_else(|| "blue".to_owned()),
            ));
        }

        Ok(results)
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    pub fn prompt(&self) -> &PromptTemplate {
        &self.prompt
    }
}
