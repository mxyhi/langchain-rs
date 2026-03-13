pub mod tools;

use langchain_core::language_models::{BaseChatModel, BaseLLM};
use langchain_core::messages::BaseMessage;
use langchain_core::runnables::RunnableConfig;

use crate::{
    EmbeddingsUnderTest, assert_chat_model_response, assert_document_embeddings_count,
    assert_embedding_dimension, assert_llm_generate_texts, assert_llm_invoke_response,
    assert_llm_token_usage, assert_query_and_documents_share_dimension, assert_usage_tokens,
};

pub trait ChatModelUnitHarness {
    type Model: BaseChatModel;

    fn model(&self) -> Self::Model;

    fn prompt(&self) -> Vec<BaseMessage>;

    fn expected_response(&self) -> &'static str;

    fn expected_usage_tokens(&self) -> Option<(usize, usize)> {
        None
    }
}

pub struct ChatModelUnitTests<H> {
    harness: H,
}

impl<H> ChatModelUnitTests<H> {
    pub fn new(harness: H) -> Self {
        Self { harness }
    }
}

impl<H> ChatModelUnitTests<H>
where
    H: ChatModelUnitHarness,
{
    pub async fn run(&self) {
        let model = self.harness.model();
        let prompt = self.harness.prompt();

        assert_chat_model_response(&model, prompt.clone(), self.harness.expected_response()).await;

        if let Some((input_tokens, output_tokens)) = self.harness.expected_usage_tokens() {
            let generated = model
                .generate(prompt, RunnableConfig::default())
                .await
                .expect("chat model unit harness should generate a response");
            assert_usage_tokens(generated.usage_metadata(), input_tokens, output_tokens)
                .expect("chat model unit harness should validate usage metadata");
        }
    }
}

pub use tools::{ToolUnitHarness, ToolsUnitTests};

pub trait EmbeddingsUnitHarness {
    type Embeddings: EmbeddingsUnderTest;

    fn embeddings(&self) -> Self::Embeddings;

    fn query(&self) -> &'static str;

    fn documents(&self) -> Vec<String>;

    fn expected_dimension(&self) -> usize;
}

pub struct EmbeddingsUnitTests<H> {
    harness: H,
}

impl<H> EmbeddingsUnitTests<H> {
    pub fn new(harness: H) -> Self {
        Self { harness }
    }
}

impl<H> EmbeddingsUnitTests<H>
where
    H: EmbeddingsUnitHarness,
{
    pub async fn run(&self) {
        let embeddings = self.harness.embeddings();
        let documents = self.harness.documents();

        assert_embedding_dimension(
            &embeddings,
            self.harness.query(),
            self.harness.expected_dimension(),
        )
        .await;
        assert_document_embeddings_count(&embeddings, documents.clone()).await;
        assert_query_and_documents_share_dimension(&embeddings, self.harness.query(), documents)
            .await;
    }
}

pub trait LlmUnitHarness {
    type Model: BaseLLM;

    fn model(&self) -> Self::Model;

    fn prompt(&self) -> &'static str;

    fn expected_response(&self) -> &'static str;

    fn prompts(&self) -> Vec<String>;

    fn expected_texts(&self) -> Vec<&'static str>;

    fn expected_token_usage(&self) -> Option<(usize, usize)> {
        None
    }
}

pub struct LlmUnitTests<H> {
    harness: H,
}

impl<H> LlmUnitTests<H> {
    pub fn new(harness: H) -> Self {
        Self { harness }
    }
}

impl<H> LlmUnitTests<H>
where
    H: LlmUnitHarness,
{
    pub async fn run(&self) {
        let model = self.harness.model();

        assert_llm_invoke_response(
            &model,
            self.harness.prompt(),
            self.harness.expected_response(),
        )
        .await;

        let expected_texts = self.harness.expected_texts();
        let result =
            assert_llm_generate_texts(&model, self.harness.prompts(), &expected_texts).await;

        if let Some((prompt_tokens, completion_tokens)) = self.harness.expected_token_usage() {
            assert_llm_token_usage(result.llm_output(), prompt_tokens, completion_tokens)
                .expect("llm unit harness should validate token usage");
        }
    }
}
