use langchain_core::embeddings::Embeddings;
use langchain_core::language_models::{BaseChatModel, BaseLLM};
use langchain_huggingface::{
    ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings,
    HuggingFacePipeline,
};

#[tokio::test]
async fn huggingface_boundaries_expose_reference_names_and_fail_honestly() {
    let chat = ChatHuggingFace::from_model_id("meta-llama/Llama-3.1-8B-Instruct");
    let endpoint = HuggingFaceEndpoint::new("meta-llama/Llama-3.1-8B-Instruct")
        .with_inference_server_url("http://localhost:8010");
    let pipeline = HuggingFacePipeline::new("meta-llama/Llama-3.1-8B-Instruct");
    let embeddings = HuggingFaceEmbeddings::new("sentence-transformers/all-mpnet-base-v2");
    let endpoint_embeddings = HuggingFaceEndpointEmbeddings::new("http://localhost:8080");

    assert_eq!(chat.model_name(), "meta-llama/Llama-3.1-8B-Instruct");
    assert_eq!(endpoint.model_name(), "meta-llama/Llama-3.1-8B-Instruct");
    assert_eq!(pipeline.model_name(), "meta-llama/Llama-3.1-8B-Instruct");
    assert_eq!(
        endpoint
            .inference_server_url()
            .expect("inference server url should exist"),
        "http://localhost:8010"
    );
    assert_eq!(endpoint_embeddings.inference_server_url(), "http://localhost:8080");

    assert!(chat
        .generate(Vec::new(), Default::default())
        .await
        .expect_err("chat transport should be unsupported")
        .to_string()
        .contains("not implemented yet"));
    assert!(endpoint
        .generate(vec!["ping".to_owned()], Default::default())
        .await
        .expect_err("endpoint transport should be unsupported")
        .to_string()
        .contains("not implemented yet"));
    assert!(pipeline
        .generate(vec!["ping".to_owned()], Default::default())
        .await
        .expect_err("pipeline transport should be unsupported")
        .to_string()
        .contains("not implemented yet"));
    assert!(embeddings
        .embed_query("ping")
        .await
        .expect_err("embeddings transport should be unsupported")
        .to_string()
        .contains("not implemented yet"));
}
