use langchain_core::load::{Reviver, SerializedValue, dumpd, dumps, load, loads};
use langchain_core::messages::HumanMessage;
use serde_json::{Map, Value, json};

fn kwargs(entries: [(&str, Value); 1]) -> Map<String, Value> {
    entries
        .into_iter()
        .map(|(key, value)| (key.to_owned(), value))
        .collect()
}

#[test]
fn dumpd_emits_python_like_constructor_shape() {
    let serialized = dumpd(&HumanMessage::new("hello")).expect("dumpd should serialize");

    assert_eq!(
        serialized,
        SerializedValue::constructor(
            vec![
                "langchain_core".to_owned(),
                "messages".to_owned(),
                "human".to_owned(),
                "HumanMessage".to_owned(),
            ],
            kwargs([("content", json!("hello"))]),
        )
    );
}

#[test]
fn loads_round_trips_registered_core_types() {
    let reviver = Reviver::core();
    let text = dumps(&HumanMessage::new("hi"), false).expect("dumps should serialize");
    let loaded = loads(&text, &reviver).expect("loads should revive registered ids");
    let message = loaded
        .as_any()
        .downcast_ref::<HumanMessage>()
        .expect("revived type should be HumanMessage");

    assert_eq!(message.content(), "hi");
}

#[test]
fn load_rejects_ids_outside_allowlist() {
    let error = load(
        SerializedValue::constructor(
            vec![
                "custom".to_owned(),
                "messages".to_owned(),
                "human".to_owned(),
                "HumanMessage".to_owned(),
            ],
            kwargs([("content", json!("blocked"))]),
        ),
        &Reviver::core(),
    )
    .expect_err("unknown ids should be rejected");

    assert!(error.to_string().contains("not allowed"));
}

#[test]
fn load_supports_legacy_ids_via_alias_mapping() {
    let loaded = load(
        SerializedValue::constructor(
            vec![
                "langchain".to_owned(),
                "schema".to_owned(),
                "messages".to_owned(),
                "HumanMessage".to_owned(),
            ],
            kwargs([("content", json!("legacy"))]),
        ),
        &Reviver::core(),
    )
    .expect("legacy ids should map to the registered type");
    let message = loaded
        .as_any()
        .downcast_ref::<HumanMessage>()
        .expect("revived type should be HumanMessage");

    assert_eq!(message.content(), "legacy");
}
