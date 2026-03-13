use std::collections::{BTreeMap, HashSet};

use futures_util::StreamExt;
use secrecy::ExposeSecret;
use serde_json::json;

use langchain_core::utils::_merge::{merge_dicts, merge_lists, merge_obj};
use langchain_core::utils::aiter::abatch_iterate;
use langchain_core::utils::env::{env_var_is_set, get_from_dict_or_env, get_from_env};
use langchain_core::utils::iter::batch_iterate;
use langchain_core::utils::strings::{
    comma_list, sanitize_for_postgres, stringify_dict, stringify_value,
};
use langchain_core::utils::usage::{add_usage, subtract_usage};
use langchain_core::utils::utils::{build_extra_kwargs, ensure_id, from_env, secret_from_env};
use langchain_core::utils::uuid::uuid7;

#[test]
fn env_helpers_resolve_values_from_dict_or_environment() {
    let env_key = format!("LANGCHAIN_RS_TEST_ENV_{}", std::process::id());
    let missing_key = format!("LANGCHAIN_RS_TEST_MISSING_{}", std::process::id());
    unsafe {
        std::env::set_var(&env_key, "from-env");
        std::env::set_var(&missing_key, "false");
    }

    let mut values = BTreeMap::new();
    values.insert("primary".to_owned(), "from-dict".to_owned());

    assert_eq!(
        get_from_dict_or_env(&values, &["primary", "secondary"], &env_key, None).unwrap(),
        "from-dict"
    );
    assert_eq!(get_from_env("api_key", &env_key, None).unwrap(), "from-env");
    assert!(env_var_is_set(&env_key));
    assert!(!env_var_is_set(&missing_key));
}

#[test]
fn string_helpers_render_nested_values_and_strip_nul_bytes() {
    let nested = json!({
        "name": "langchain",
        "tags": ["rust", "parity"],
        "meta": { "version": 1 }
    });

    assert!(stringify_value(&nested).contains("name: langchain"));
    assert!(
        stringify_dict(nested.as_object().expect("json object")).contains("tags: rust\nparity")
    );
    assert_eq!(comma_list(["alpha", "beta", "gamma"]), "alpha, beta, gamma");
    assert_eq!(sanitize_for_postgres("a\u{0}b", " "), "a b");
}

#[test]
fn usage_and_merge_helpers_combine_nested_json_payloads() {
    let combined = add_usage(
        json!({
            "input_tokens": 3,
            "nested": { "prompt_tokens": 2 }
        })
        .as_object()
        .expect("json object"),
        json!({
            "input_tokens": 4,
            "nested": { "prompt_tokens": 5 }
        })
        .as_object()
        .expect("json object"),
    )
    .unwrap();
    assert_eq!(combined["input_tokens"], 7);
    assert_eq!(combined["nested"]["prompt_tokens"], 7);

    let subtracted = subtract_usage(
        json!({
            "input_tokens": 7,
            "nested": { "prompt_tokens": 9 }
        })
        .as_object()
        .expect("json object"),
        json!({
            "input_tokens": 4,
            "nested": { "prompt_tokens": 5 }
        })
        .as_object()
        .expect("json object"),
    )
    .unwrap();
    assert_eq!(subtracted["input_tokens"], 3);
    assert_eq!(subtracted["nested"]["prompt_tokens"], 4);

    let merged_dict = merge_dicts(
        json!({
            "name": "tool-",
            "count": 1,
            "nested": { "value": "hel" },
            "items": [{ "index": 0, "text": "foo" }]
        })
        .as_object()
        .expect("json object"),
        [json!({
            "name": "call",
            "count": 2,
            "nested": { "value": "lo" },
            "items": [{ "index": 0, "text": "bar" }, { "index": 1, "text": "baz" }]
        })
        .as_object()
        .expect("json object")],
    )
    .unwrap();
    assert_eq!(merged_dict["name"], "tool-call");
    assert_eq!(merged_dict["count"], 3);
    assert_eq!(merged_dict["nested"]["value"], "hello");
    assert_eq!(merged_dict["items"][0]["text"], "foobar");
    assert_eq!(merged_dict["items"][1]["text"], "baz");

    let merged_list = merge_lists(
        &[json!({"index": 0, "name": "foo"})],
        [&vec![json!({"index": 0, "name": "-bar"}), json!("tail")]],
    )
    .unwrap();
    assert_eq!(merged_list[0]["name"], "foo-bar");
    assert_eq!(merged_list[1], "tail");

    let merged_obj = merge_obj(&json!({"value": ["a"]}), &json!({"value": ["b"]})).unwrap();
    assert_eq!(merged_obj["value"], json!(["a", "b"]));
}

#[test]
fn iterator_and_uuid_helpers_expand_utils_surface() {
    let batches = batch_iterate(Some(2), 1..=5).collect::<Vec<_>>();
    assert_eq!(batches, vec![vec![1, 2], vec![3, 4], vec![5]]);

    let generated = (0..8).map(|_| uuid7()).collect::<Vec<_>>();
    assert!(generated.iter().all(|uuid| uuid.get_version_num() == 7));

    let ids = generated.into_iter().collect::<HashSet<_>>();
    assert_eq!(ids.len(), 8);
}

#[tokio::test]
async fn async_batch_iterate_and_lazy_env_factories_work() {
    let batches = abatch_iterate(Some(2), futures_util::stream::iter(vec![1, 2, 3, 4, 5]))
        .collect::<Vec<_>>()
        .await;
    assert_eq!(batches, vec![vec![1, 2], vec![3, 4], vec![5]]);

    let env_key = format!("LANGCHAIN_RS_TEST_FACTORY_{}", std::process::id());
    unsafe {
        std::env::set_var(&env_key, "factory-value");
    }

    let loader = from_env(env_key.clone(), None, None);
    assert_eq!(loader().unwrap(), "factory-value");

    let secret_loader = secret_from_env(env_key, None, None);
    assert_eq!(secret_loader().unwrap().expose_secret(), "factory-value");
}

#[test]
fn utility_helpers_build_extra_kwargs_and_ids() {
    let extra = build_extra_kwargs(
        BTreeMap::from([("temperature".to_owned(), json!(0.2))]),
        BTreeMap::from([
            ("model".to_owned(), json!("gpt-4o-mini")),
            ("timeout".to_owned(), json!(30)),
        ]),
        ["model".to_owned()].into_iter().collect(),
    )
    .unwrap();

    assert_eq!(extra["temperature"], json!(0.2));
    assert_eq!(extra["timeout"], json!(30));

    let preserved = ensure_id(Some("tool-call-1".to_owned()));
    assert_eq!(preserved, "tool-call-1");
    assert!(ensure_id(None).starts_with("lc_"));
}
