use langchain_core::language_models::BaseLLM;
use langchain_core::outputs::GenerationCandidate;
use langchain_core::runnables::Runnable;
use langchain_openai::OpenAI;
use serde_json::json;
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn invokes_completions_and_maps_llm_result() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/completions"))
        .and(header("authorization", "Bearer test-key"))
        .and(body_json(json!({
            "model": "gpt-3.5-turbo-instruct",
            "prompt": ["alpha", "beta"]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "cmpl_test",
            "model": "gpt-3.5-turbo-instruct",
            "system_fingerprint": "fp_test",
            "choices": [
                {
                    "index": 0,
                    "text": "ALPHA",
                    "finish_reason": "stop",
                    "logprobs": null
                },
                {
                    "index": 1,
                    "text": "BETA",
                    "finish_reason": "length",
                    "logprobs": { "tokens": ["BETA"] }
                }
            ],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 9,
                "total_tokens": 18
            }
        })))
        .mount(&server)
        .await;

    let model = OpenAI::new("gpt-3.5-turbo-instruct", server.uri(), Some("test-key"));
    let result = model
        .generate(
            vec!["alpha".to_owned(), "beta".to_owned()],
            Default::default(),
        )
        .await
        .expect("llm generate should succeed");

    assert_eq!(result.generations().len(), 2);
    assert_eq!(result.generations()[0][0].text(), "ALPHA");
    assert_eq!(result.generations()[1][0].text(), "BETA");
    assert!(matches!(
        &result.generations()[1][0],
        GenerationCandidate::Text(generation)
            if generation
                .generation_info()
                .and_then(|info| info.get("finish_reason"))
                == Some(&json!("length"))
    ));
    assert_eq!(
        result
            .llm_output()
            .and_then(|output| output.get("model_name")),
        Some(&json!("gpt-3.5-turbo-instruct"))
    );
    assert_eq!(
        result
            .llm_output()
            .and_then(|output| output.get("system_fingerprint")),
        Some(&json!("fp_test"))
    );
}

#[tokio::test]
async fn openai_llm_is_runnable_for_single_prompt() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "cmpl_single",
            "model": "gpt-3.5-turbo-instruct",
            "choices": [
                {
                    "index": 0,
                    "text": "pong",
                    "finish_reason": "stop",
                    "logprobs": null
                }
            ]
        })))
        .mount(&server)
        .await;

    let model = OpenAI::new("gpt-3.5-turbo-instruct", server.uri(), None::<&str>);
    let output = model
        .invoke("ping".to_owned(), Default::default())
        .await
        .expect("llm invoke should succeed");

    assert_eq!(output, "pong");
}

#[tokio::test]
async fn returns_http_status_error_for_failed_completions_request() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/completions"))
        .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
        .mount(&server)
        .await;

    let model = OpenAI::new("gpt-3.5-turbo-instruct", server.uri(), None::<&str>);
    let error = model
        .generate(vec!["ping".to_owned()], Default::default())
        .await
        .expect_err("failed request should bubble up");

    assert_eq!(
        error.to_string(),
        "upstream returned http 429: rate limited"
    );
}
