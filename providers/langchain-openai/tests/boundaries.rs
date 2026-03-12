use langchain_core::embeddings::Embeddings;
use langchain_core::language_models::{BaseChatModel, BaseLLM};
use langchain_core::messages::BaseMessage;
use langchain_openai::{AzureChatOpenAI, AzureOpenAI, AzureOpenAIEmbeddings, custom_tool};
use serde_json::json;

#[tokio::test]
async fn azure_boundaries_exist_and_fail_honestly() {
    let chat = AzureChatOpenAI::new(
        "gpt-4o-mini",
        "test-deployment",
        "https://example-resource.openai.azure.com",
        Some("test-key"),
    );
    let llm = AzureOpenAI::new(
        "gpt-35-turbo-instruct",
        "test-deployment",
        "https://example-resource.openai.azure.com",
        Some("test-key"),
    );
    let embeddings = AzureOpenAIEmbeddings::new(
        "text-embedding-3-small",
        "test-deployment",
        "https://example-resource.openai.azure.com",
        Some("test-key"),
    );

    assert_eq!(chat.model_name(), "gpt-4o-mini");
    assert_eq!(llm.model_name(), "gpt-35-turbo-instruct");
    assert_eq!(embeddings.model_name(), "text-embedding-3-small");
    assert_eq!(chat.deployment_name(), "test-deployment");
    assert_eq!(llm.deployment_name(), "test-deployment");
    assert_eq!(embeddings.deployment_name(), "test-deployment");
    assert_eq!(
        chat.base_url(),
        "https://example-resource.openai.azure.com/openai/deployments/test-deployment"
    );

    assert!(
        BaseChatModel::generate(&chat, Vec::<BaseMessage>::new(), Default::default())
            .await
            .expect_err("azure chat boundary should be explicit unsupported")
            .to_string()
            .contains("AzureChatOpenAI transport is not implemented")
    );
    assert!(
        BaseLLM::generate(&llm, vec!["ping".to_owned()], Default::default())
            .await
            .expect_err("azure llm boundary should be explicit unsupported")
            .to_string()
            .contains("AzureOpenAI transport is not implemented")
    );
    assert!(
        embeddings
            .embed_query("ping")
            .await
            .expect_err("azure embeddings boundary should be explicit unsupported")
            .to_string()
            .contains("AzureOpenAIEmbeddings transport is not implemented")
    );
}

#[test]
fn custom_tool_exposes_openai_boundary_metadata() {
    let tool = custom_tool("bash_execute", "Run a shell command")
        .with_parameters(json!({
            "type": "object",
            "properties": {
                "command": { "type": "string" }
            },
            "required": ["command"]
        }))
        .with_format(json!({
            "type": "grammar",
            "syntax": "lark",
            "definition": "start: WORD"
        }));

    assert_eq!(tool.name(), "bash_execute");
    assert_eq!(tool.description(), "Run a shell command");
    assert_eq!(tool.metadata(), &json!({ "type": "custom_tool" }));
    assert_eq!(
        tool.parameters(),
        &json!({
            "type": "object",
            "properties": {
                "command": { "type": "string" }
            },
            "required": ["command"]
        })
    );
    assert_eq!(
        tool.format(),
        Some(&json!({
            "type": "grammar",
            "syntax": "lark",
            "definition": "start: WORD"
        }))
    );
}
