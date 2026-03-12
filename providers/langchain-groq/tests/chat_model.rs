use langchain_groq::ChatGroq;

#[test]
fn groq_uses_reference_default_base_url() {
    let model = ChatGroq::new("llama-3.1-8b-instant", None::<&str>);
    assert_eq!(model.base_url(), "https://api.groq.com/openai/v1");
}
