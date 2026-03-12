use std::collections::BTreeMap;

use langchain_core::LangChainError;
use langchain_core::language_models::BaseLLM;
use langchain_core::prompts::{PromptArgument, PromptArguments, PromptTemplate};
use langchain_core::runnables::RunnableConfig;

pub const TEST_GEN_TEMPLATE_SUFFIX: &str = "Add another example.";

pub async fn generate_example(
    examples: &[BTreeMap<String, String>],
    llm: &dyn BaseLLM,
    prompt_template: &PromptTemplate,
) -> Result<String, LangChainError> {
    let mut rendered_examples = Vec::with_capacity(examples.len() + 1);

    for example in examples {
        let arguments = example
            .iter()
            .map(|(name, value)| (name.clone(), PromptArgument::String(value.clone())))
            .collect::<PromptArguments>();
        rendered_examples.push(prompt_template.format(&arguments)?);
    }

    rendered_examples.push(TEST_GEN_TEMPLATE_SUFFIX.to_owned());
    llm.invoke_prompt(rendered_examples.join("\n\n"), RunnableConfig::default())
        .await
}
