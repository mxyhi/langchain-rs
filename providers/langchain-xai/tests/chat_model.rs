use langchain_xai::ChatXAI;

#[test]
fn xai_uses_reference_default_base_url() {
    let model = ChatXAI::new("grok-3-mini", None::<&str>);
    assert_eq!(model.base_url(), "https://api.x.ai/v1");
}
