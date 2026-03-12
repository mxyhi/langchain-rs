use langchain_core::language_models::BaseLLM;

#[test]
fn huggingface_namespaces_match_root_exports() {
    let chat = langchain_huggingface::chat_models::ChatHuggingFace::from_model_id(
        "meta-llama/Llama-3.1-8B-Instruct",
    );
    assert_eq!(chat.model_id(), "meta-llama/Llama-3.1-8B-Instruct");

    let embeddings = langchain_huggingface::embeddings::HuggingFaceEmbeddings::new(
        "sentence-transformers/all-mpnet-base-v2",
    );
    assert_eq!(
        embeddings.model_name(),
        "sentence-transformers/all-mpnet-base-v2"
    );

    let endpoint_embeddings = langchain_huggingface::embeddings::HuggingFaceEndpointEmbeddings::new(
        "http://localhost:8080",
    );
    assert_eq!(
        endpoint_embeddings.inference_server_url(),
        "http://localhost:8080"
    );

    let endpoint =
        langchain_huggingface::llms::HuggingFaceEndpoint::new("mistralai/Mistral-7B-Instruct-v0.3")
            .with_inference_server_url("http://localhost:8081");
    assert_eq!(
        endpoint.inference_server_url(),
        Some("http://localhost:8081")
    );

    let pipeline =
        langchain_huggingface::llms::HuggingFacePipeline::new("meta-llama/Llama-3.1-8B-Instruct");
    assert_eq!(pipeline.model_name(), "meta-llama/Llama-3.1-8B-Instruct");
}
