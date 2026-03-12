use langchain_openrouter::ChatOpenRouter;

#[test]
fn openrouter_uses_reference_default_base_url() {
    let model = ChatOpenRouter::new("openai/gpt-4o-mini", None::<&str>);
    assert_eq!(model.base_url(), "https://openrouter.ai/api/v1");
}
