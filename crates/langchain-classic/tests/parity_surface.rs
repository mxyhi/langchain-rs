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
use langchain_core::language_models::{ParrotChatModel, ParrotLLM};
use langchain_core::messages::HumanMessage;
use langchain_core::prompts::{PromptArgument, PromptArguments, PromptTemplate};
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

#[tokio::test]
async fn classic_boundary_modules_expose_real_surface() {
    let message = HumanMessage::new("hello classic").into();
    let openai_dict = langchain_classic::adapters::openai::convert_message_to_dict(&message);
    assert_eq!(openai_dict["role"], "user");
    assert_eq!(openai_dict["content"], "hello classic");

    let round_tripped =
        langchain_classic::adapters::openai::convert_openai_messages(vec![openai_dict.clone()])
            .expect("openai adapter conversion should succeed");
    assert_eq!(round_tripped, vec![message.clone()]);

    let html_docs = vec![
        Document::new("<article><h1>Alpha</h1><p>Beta</p></article>"),
        Document::new("<div>Gamma</div>"),
    ];
    let html2text = langchain_classic::document_transformers::Html2TextTransformer::new();
    let plain_docs = html2text.transform_documents(html_docs.clone());
    assert_eq!(plain_docs[0].page_content, "Alpha Beta");

    let reordered = langchain_classic::document_transformers::LongContextReorder::new()
        .transform_documents(vec![
            Document::new("first"),
            Document::new("second"),
            Document::new("third"),
            Document::new("fourth"),
        ]);
    assert_eq!(
        reordered
            .iter()
            .map(|document| document.page_content.as_str())
            .collect::<Vec<_>>(),
        vec!["first", "fourth", "second", "third"]
    );

    let evaluator = langchain_classic::evaluation::load_evaluator(
        langchain_classic::evaluation::EvaluatorType::ExactMatch,
    )
    .expect("exact-match evaluator should load");
    let evaluation = evaluator.evaluate_strings("same", Some("same"), None);
    assert_eq!(evaluation.score, Some(1.0));
    assert_eq!(evaluation.value, Some(true));

    let graph = langchain_classic::graphs::Neo4jGraph::new("bolt://localhost:7687");
    assert_eq!(graph.provider_name(), "neo4j");
    assert_eq!(graph.uri(), "bolt://localhost:7687");
    assert!(!graph.is_remote_connected());

    let runs = langchain_classic::smith::run_on_dataset(
        &[
            langchain_classic::smith::DatasetExample::new("same", Some("same")),
            langchain_classic::smith::DatasetExample::new("same", Some("different")),
        ],
        &langchain_classic::smith::RunEvalConfig::default()
            .with_evaluator(langchain_classic::evaluation::EvaluatorType::ExactMatch),
    )
    .expect("smith dataset runner should evaluate rows");
    assert_eq!(runs.summary().total_examples, 2);
    assert_eq!(runs.summary().successful_evaluations, 2);
    assert_eq!(runs.summary().average_score, Some(0.5));
}

#[tokio::test]
async fn classic_root_reexports_legacy_names_without_faking_support() {
    let mrkl = langchain_classic::MRKLChain::new(ParrotChatModel::new("classic-agent", 12));
    let react = langchain_classic::ReActChain::new(ParrotChatModel::new("classic-agent", 12));
    let self_ask =
        langchain_classic::SelfAskWithSearchChain::new(ParrotChatModel::new("classic-agent", 12));

    assert_eq!(mrkl.strategy_name(), "mrkl");
    assert_eq!(react.strategy_name(), "react");
    assert_eq!(self_ask.strategy_name(), "self-ask-with-search");

    let prompt = PromptTemplate::new("Answer {input}");
    let arguments: PromptArguments = [(
        "input".to_owned(),
        PromptArgument::String("ping".to_owned()),
    )]
    .into();

    let checker =
        langchain_classic::LLMCheckerChain::new(ParrotLLM::new("classic-llm", 32), prompt.clone());
    let math =
        langchain_classic::LLMMathChain::new(ParrotLLM::new("classic-llm", 32), prompt.clone());
    let qa = langchain_classic::QAWithSourcesChain::new(
        ParrotLLM::new("classic-llm", 32),
        prompt.clone(),
    );
    let vectordb =
        langchain_classic::VectorDBQA::new(ParrotLLM::new("classic-llm", 32), prompt.clone());
    let vectordb_sources = langchain_classic::VectorDBQAWithSourcesChain::new(
        ParrotLLM::new("classic-llm", 32),
        prompt,
    );

    assert!(checker.purpose().contains("LLM checker"));
    assert!(math.purpose().contains("math"));
    assert!(qa.purpose().contains("sources"));
    assert!(vectordb.purpose().contains("vector"));
    assert!(vectordb_sources.purpose().contains("sources"));

    assert!(
        checker
            .run(arguments.clone())
            .await
            .expect("checker chain should run")
            .contains("Answer ping")
    );
    assert!(
        math.run(arguments.clone())
            .await
            .expect("math chain should run")
            .contains("Answer ping")
    );
    assert!(
        qa.run(arguments.clone())
            .await
            .expect("qa chain should run")
            .contains("Answer ping")
    );
    assert!(
        vectordb
            .run(arguments.clone())
            .await
            .expect("vectordb chain should run")
            .contains("Answer ping")
    );
    assert!(
        vectordb_sources
            .run(arguments)
            .await
            .expect("vectordb-with-sources chain should run")
            .contains("Answer ping")
    );
}
