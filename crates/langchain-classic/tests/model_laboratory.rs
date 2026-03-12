use std::collections::BTreeMap;

use langchain_classic::example_generator::{TEST_GEN_TEMPLATE_SUFFIX, generate_example};
use langchain_classic::llms::ParrotLLM;
use langchain_classic::model_laboratory::ModelLaboratory;
use langchain_classic::prompts::PromptTemplate;

#[tokio::test]
async fn generate_example_renders_examples_and_suffix_before_calling_llm() {
    let examples = vec![BTreeMap::from([
        ("question".to_owned(), "2 + 2".to_owned()),
        ("answer".to_owned(), "4".to_owned()),
    ])];
    let llm = ParrotLLM::new("example-generator", 256);
    let prompt = PromptTemplate::new("Question: {question}\nAnswer: {answer}");

    let rendered = generate_example(&examples, &llm, &prompt)
        .await
        .expect("example generation should invoke llm");

    assert!(rendered.contains("Question: 2 + 2"));
    assert!(rendered.contains("Answer: 4"));
    assert!(rendered.contains(TEST_GEN_TEMPLATE_SUFFIX));
}

#[tokio::test]
async fn model_laboratory_compares_multiple_llms() {
    let lab = ModelLaboratory::from_llms(
        vec![
            Box::new(ParrotLLM::new("lab-a", 64)),
            Box::new(ParrotLLM::new("lab-b", 64)),
        ],
        Some(PromptTemplate::new("Prompt: {_input}")),
    )
    .expect("model laboratory should initialize");

    let results = lab
        .compare("hello world")
        .await
        .expect("model laboratory should compare outputs");

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].name(), "lab-a");
    assert_eq!(results[1].name(), "lab-b");
    assert!(results[0].output().contains("Prompt: hello world"));
    assert!(!results[0].color().is_empty());
}
