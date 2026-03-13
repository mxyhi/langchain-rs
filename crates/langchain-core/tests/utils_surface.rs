use std::collections::BTreeMap;

use langchain_core::utils::{
    _merge::{merge_dicts, merge_lists, merge_obj},
    env::{env_var_is_set, get_from_dict_or_env, get_from_env},
    strings::{comma_list, sanitize_for_postgres, stringify_dict, stringify_value},
    usage::{add_usage, subtract_usage},
    uuid::uuid7,
};
use serde_json::{Map, Value, json};
use uuid::Version;

fn object(value: Value) -> Map<String, Value> {
    value.as_object().cloned().expect("expected object")
}

#[test]
fn env_helpers_resolve_dict_env_and_default_values() {
    // SAFETY: these tests use a unique variable name and restore process env before exit.
    unsafe {
        std::env::remove_var("LANGCHAIN_CORE_UTILS_TEST_TOKEN");
    }
    assert!(!env_var_is_set("LANGCHAIN_CORE_UTILS_TEST_TOKEN"));

    let config = BTreeMap::from([
        ("secondary".to_owned(), "from-config".to_owned()),
        ("empty".to_owned(), String::new()),
    ]);
    assert_eq!(
        get_from_dict_or_env(
            &config,
            ["primary", "secondary"],
            "LANGCHAIN_CORE_UTILS_TEST_TOKEN",
            None
        )
        .expect("config value should win"),
        "from-config"
    );

    // SAFETY: these tests use a unique variable name and restore process env before exit.
    unsafe {
        std::env::set_var("LANGCHAIN_CORE_UTILS_TEST_TOKEN", "from-env");
    }
    assert!(env_var_is_set("LANGCHAIN_CORE_UTILS_TEST_TOKEN"));
    assert_eq!(
        get_from_env("token", "LANGCHAIN_CORE_UTILS_TEST_TOKEN", None)
            .expect("env value should exist"),
        "from-env"
    );
    assert_eq!(
        get_from_dict_or_env(
            &BTreeMap::new(),
            ["primary", "secondary"],
            "LANGCHAIN_CORE_UTILS_TEST_TOKEN",
            Some("fallback")
        )
        .expect("env value should win over default"),
        "from-env"
    );

    // SAFETY: these tests use a unique variable name and restore process env before exit.
    unsafe {
        std::env::remove_var("LANGCHAIN_CORE_UTILS_TEST_TOKEN");
    }
    assert_eq!(
        get_from_env("token", "LANGCHAIN_CORE_UTILS_TEST_TOKEN", Some("fallback"))
            .expect("default value should be used"),
        "fallback"
    );
}

#[test]
fn string_helpers_match_reference_shape() {
    let value = json!({
        "name": "LangChain",
        "tags": ["rust", "utils"],
        "nested": {"level": 1}
    });
    let map = value.as_object().expect("json object");

    assert_eq!(stringify_value(&json!("plain text")), "plain text");
    assert!(stringify_value(&value).contains("name: LangChain"));
    assert!(stringify_dict(map).contains("tags: rust\nutils"));
    assert_eq!(comma_list(["alpha", "beta", "gamma"]), "alpha, beta, gamma");
    assert_eq!(sanitize_for_postgres("hi\0there", " "), "hi there");
}

#[test]
fn merge_helpers_merge_nested_dicts_lists_and_strings() {
    let left = object(json!({
        "content": "hel",
        "tool_calls": [{"index": 0, "id": "call_1", "args": "{"}],
        "metadata": {"count": 1, "optional": null}
    }));
    let right = object(json!({
        "content": "lo",
        "tool_calls": [{"index": 0, "id": "call_1", "args": "\"x\":1}"}],
        "metadata": {"count": 2, "optional": "done"}
    }));

    let merged = merge_dicts(&left, [&right]).expect("dicts should merge");
    assert_eq!(merged["content"], json!("hello"));
    assert_eq!(merged["metadata"]["count"], json!(3));
    assert_eq!(merged["metadata"]["optional"], json!("done"));
    assert_eq!(merged["tool_calls"][0]["args"], json!("{\"x\":1}"));

    let merged_list = merge_lists(
        left["tool_calls"].as_array().expect("left list"),
        [right["tool_calls"].as_array().expect("right list")],
    )
    .expect("lists should merge");
    assert_eq!(merged_list.len(), 1);
    assert_eq!(merged_list[0]["args"], json!("{\"x\":1}"));

    assert_eq!(
        merge_obj(&json!({"key": "va"}), &json!({"key": "lue"})).expect("objects"),
        json!({"key": "value"})
    );
}

#[test]
fn usage_helpers_apply_recursive_integer_math() {
    let left = object(json!({
        "prompt_tokens": 10,
        "details": {"cache_read": 4, "accepted": 2}
    }));
    let right = object(json!({
        "prompt_tokens": 3,
        "details": {"cache_read": 1, "accepted": 1}
    }));

    let added = add_usage(&left, &right).expect("usage should add");
    assert_eq!(added["prompt_tokens"], json!(13));
    assert_eq!(added["details"]["cache_read"], json!(5));

    let subtracted = subtract_usage(&left, &right).expect("usage should subtract");
    assert_eq!(subtracted["prompt_tokens"], json!(7));
    assert_eq!(subtracted["details"]["accepted"], json!(1));
}

#[test]
fn uuid7_helper_exposes_sortable_version7_ids() {
    let first = uuid7();
    let second = uuid7();

    assert_eq!(first.get_version(), Some(Version::SortRand));
    assert_eq!(second.get_version(), Some(Version::SortRand));
    assert!(first <= second);
}
