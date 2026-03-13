use std::collections::BTreeMap;

use langchain_classic::docstore::{DocstoreFn, InMemoryDocstore};
use langchain_classic::storage::{
    BaseStore, EncoderBackedStore, InMemoryByteStore, LocalFileStore, create_kv_docstore,
    create_lc_store,
};
use langchain_classic::utils::{
    comma_list, env_var_is_set, get_from_dict_or_env, get_from_env, sanitize_for_postgres,
    stringify_dict, stringify_value,
};
use langchain_core::documents::Document;
use serde_json::{Map, json};
use tempfile::tempdir;

#[test]
fn classic_utils_reexport_core_utility_helpers() {
    let map = Map::from_iter([("city".to_owned(), json!("Shanghai"))]);
    let env_key = "LANGCHAIN_CLASSIC_UTILS_ENV";

    unsafe {
        std::env::remove_var(env_key);
        std::env::set_var(env_key, "from-env");
    }

    assert_eq!(comma_list(["alpha", "beta", "gamma"]), "alpha, beta, gamma");
    assert_eq!(stringify_dict(&map), "city: Shanghai\n");
    assert_eq!(stringify_value(&json!(["alpha", "beta"])), "alpha\nbeta");
    assert_eq!(sanitize_for_postgres("a\0b", ""), "ab");
    assert!(env_var_is_set(env_key));
    assert_eq!(
        get_from_env("value", env_key, None).expect("env lookup should succeed"),
        "from-env"
    );
    assert_eq!(
        get_from_dict_or_env(
            &BTreeMap::from([(String::from("value"), String::from("from-dict"))]),
            ["value"],
            env_key,
            None,
        )
        .expect("dict lookup should win"),
        "from-dict"
    );
}

#[test]
fn classic_docstore_and_storage_helpers_round_trip_documents() {
    let docstore = InMemoryDocstore::new();
    let alpha = Document::new("alpha");
    docstore.add("doc-1", alpha.clone());
    assert_eq!(
        docstore
            .search("doc-1")
            .expect("docstore lookup should succeed"),
        Some(alpha.clone())
    );

    let lookup = DocstoreFn::new({
        let docstore = docstore.clone();
        move |id| docstore.search(id).ok().flatten()
    });
    assert_eq!(
        lookup
            .search("doc-1")
            .expect("function docstore should succeed"),
        Some(alpha.clone())
    );

    let bytes = InMemoryByteStore::new();
    let doc_store = create_kv_docstore(bytes);
    doc_store.mset(vec![("doc-2".to_owned(), Document::new("beta"))]);
    assert_eq!(
        doc_store.mget(&["doc-2".to_owned()]),
        vec![Some(Document::new("beta"))]
    );

    let json_store = create_lc_store(InMemoryByteStore::new());
    json_store.mset(vec![("payload".to_owned(), json!({"ok": true}))]);
    assert_eq!(
        json_store.mget(&["payload".to_owned()]),
        vec![Some(json!({"ok": true}))]
    );

    let local_dir = tempdir().expect("temp dir should exist");
    let local = LocalFileStore::new(local_dir.path());
    local.mset(vec![("folder/value".to_owned(), b"hello".to_vec())]);
    assert_eq!(
        local.mget(&["folder/value".to_owned()]),
        vec![Some(b"hello".to_vec())]
    );
    assert_eq!(
        local.yield_keys(Some("folder")).collect::<Vec<_>>(),
        vec!["folder/value".to_owned()]
    );

    fn assert_encoder_store<T>(_store: &EncoderBackedStore<InMemoryByteStore, T>) {}
    assert_encoder_store(&create_lc_store::<serde_json::Value, _>(
        InMemoryByteStore::new(),
    ));
}

#[test]
fn classic_root_and_namespace_aliases_expose_parity_surface() {
    let _python = langchain_classic::PythonREPL::new();
    let _requests = langchain_classic::Requests::new();
    let _wrapper = langchain_classic::RequestsWrapper::new();
    let _text_wrapper = langchain_classic::TextRequestsWrapper::new();
    let _sql = langchain_classic::SQLDatabase::from_sqlite_path(":memory:");
    let _serp = langchain_classic::SerpAPIWrapper::new("test-key");
    let _docstore = langchain_classic::InMemoryDocstore::new();

    assert_eq!(
        langchain_classic::_api::CLASSIC_PACKAGE,
        "langchain_classic"
    );
    assert_eq!(
        langchain_classic::adapters::CLASSIC_PACKAGE,
        "langchain_classic"
    );
    assert_eq!(
        langchain_classic::document_transformers::CLASSIC_PACKAGE,
        "langchain_classic"
    );
    assert_eq!(
        langchain_classic::evaluation::CLASSIC_PACKAGE,
        "langchain_classic"
    );
    assert_eq!(
        langchain_classic::graphs::CLASSIC_PACKAGE,
        "langchain_classic"
    );
    assert_eq!(
        langchain_classic::smith::CLASSIC_PACKAGE,
        "langchain_classic"
    );

    let manager = langchain_classic::indexes::InMemoryRecordManager::new("classic-indexes");
    assert_eq!(manager.namespace(), "classic-indexes");
}
