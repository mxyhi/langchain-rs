use langchain_openai::middleware::OpenAIModerationClient;
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn moderation_client_posts_input_and_parses_flagged_result() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/moderations"))
        .and(header("authorization", "Bearer test-key"))
        .and(body_json(serde_json::json!({
            "model": "omni-moderation-latest",
            "input": "I want to kill them."
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "modr-123",
            "model": "omni-moderation-latest",
            "results": [{
                "flagged": true,
                "categories": {
                    "harassment": true,
                    "violence": true
                },
                "category_scores": {
                    "harassment": 0.52,
                    "violence": 0.99
                },
                "category_applied_input_types": {
                    "harassment": ["text"],
                    "violence": ["text"]
                }
            }]
        })))
        .mount(&server)
        .await;

    let client = OpenAIModerationClient::new(server.uri(), Some("test-key"));
    let result = client
        .moderate_text("I want to kill them.")
        .await
        .expect("moderation request should succeed");

    assert!(result.flagged);
    assert_eq!(result.categories["harassment"], serde_json::json!(true));
    assert_eq!(result.category_scores["violence"], serde_json::json!(0.99));
    assert_eq!(
        result.category_applied_input_types["violence"],
        serde_json::json!(["text"])
    );
}

#[tokio::test]
async fn moderation_client_allows_model_override() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/moderations"))
        .and(body_json(serde_json::json!({
            "model": "text-moderation-latest",
            "input": "safe text"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "modr-456",
            "model": "text-moderation-latest",
            "results": [{
                "flagged": false,
                "categories": {},
                "category_scores": {}
            }]
        })))
        .mount(&server)
        .await;

    let client = OpenAIModerationClient::new(server.uri(), None::<&str>)
        .with_model("text-moderation-latest");
    let result = client
        .moderate_text("safe text")
        .await
        .expect("moderation request should succeed");

    assert!(!result.flagged);
    assert_eq!(client.model(), "text-moderation-latest");
}
