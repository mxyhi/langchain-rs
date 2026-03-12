use langchain_core::embeddings::Embeddings;
use langchain_core::language_models::{BaseChatModel, BaseLLM};
use langchain_core::messages::{BaseMessage, HumanMessage};
use langchain_openai::{AzureChatOpenAI, AzureOpenAI, AzureOpenAIEmbeddings, custom_tool};
use serde_json::json;
use wiremock::matchers::{header, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn azure_boundaries_route_requests_to_deployment_scoped_openai_endpoints() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/openai/deployments/test-deployment/chat/completions"))
        .and(query_param("api-version", "2024-02-01"))
        .and(header("api-key", "test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "azure-chat-1",
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "azure chat ok"
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "total_tokens": 5
            }
        })))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/openai/deployments/test-deployment/completions"))
        .and(query_param("api-version", "2024-02-01"))
        .and(header("api-key", "test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [
                { "text": "azure llm ok", "finish_reason": "stop" }
            ],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "total_tokens": 5
            },
            "model": "gpt-35-turbo-instruct"
        })))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/openai/deployments/test-deployment/embeddings"))
        .and(query_param("api-version", "2024-02-01"))
        .and(header("api-key", "test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [
                { "embedding": [0.1, 0.2, 0.3] }
            ]
        })))
        .mount(&server)
        .await;

    let chat = AzureChatOpenAI::new(
        "gpt-4o-mini",
        "test-deployment",
        server.uri(),
        Some("test-key"),
    );
    let llm = AzureOpenAI::new(
        "gpt-35-turbo-instruct",
        "test-deployment",
        server.uri(),
        Some("test-key"),
    );
    let embeddings = AzureOpenAIEmbeddings::new(
        "text-embedding-3-small",
        "test-deployment",
        server.uri(),
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
        format!("{}/openai/deployments/test-deployment", server.uri())
    );

    let chat_response = BaseChatModel::generate(
        &chat,
        vec![BaseMessage::from(HumanMessage::new("ping"))],
        Default::default(),
    )
    .await
    .expect("azure chat request should succeed");
    assert_eq!(chat_response.content(), "azure chat ok");

    let llm_response = BaseLLM::generate(&llm, vec!["ping".to_owned()], Default::default())
        .await
        .expect("azure llm request should succeed");
    assert_eq!(llm_response.generations()[0][0].text(), "azure llm ok");

    let embedding = embeddings
        .embed_query("ping")
        .await
        .expect("azure embeddings request should succeed");
    assert_eq!(embedding, vec![0.1, 0.2, 0.3]);
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
