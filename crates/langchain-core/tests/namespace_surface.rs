use std::collections::BTreeMap;

use langchain_core::agents::{
    AgentAction, AgentActionMessageLog, AgentFinish, AgentStep, convert_agent_action_to_messages,
    convert_agent_observation_to_messages,
};
use langchain_core::chat_loaders::BaseChatLoader;
use langchain_core::chat_sessions::ChatSession;
use langchain_core::env::get_runtime_environment;
use langchain_core::messages::{BaseMessage, HumanMessage};
use langchain_core::stores::{BaseStore, InMemoryByteStore, InMemoryStore};
use langchain_core::structured_query::{
    Comparator, Comparison, FilterDirective, Operation, Operator, StructuredQuery, Visitor,
};
use langchain_core::sys_info::system_info_report;
use serde_json::{Value, json};

struct StaticChatLoader;

impl BaseChatLoader for StaticChatLoader {
    fn lazy_load<'a>(&'a self) -> Box<dyn Iterator<Item = ChatSession> + 'a> {
        Box::new(
            vec![
                ChatSession::new()
                    .with_messages(vec![HumanMessage::new("hello").into()])
                    .with_functions(vec![json!({"name": "lookup"})]),
                ChatSession::new().with_messages(vec![HumanMessage::new("world").into()]),
            ]
            .into_iter(),
        )
    }
}

struct QueryPrinter;

impl Visitor<String> for QueryPrinter {
    fn visit_operation(&mut self, operation: &Operation) -> String {
        let parts = operation
            .arguments()
            .iter()
            .map(|argument| argument.accept(self))
            .collect::<Vec<_>>();
        format!("{:?}({})", operation.operator(), parts.join(","))
    }

    fn visit_comparison(&mut self, comparison: &Comparison) -> String {
        format!(
            "{:?}({}, {})",
            comparison.comparator(),
            comparison.attribute(),
            comparison.value()
        )
    }

    fn visit_structured_query(&mut self, structured_query: &StructuredQuery) -> String {
        let filter = structured_query
            .filter()
            .map(|filter| filter.accept(self))
            .unwrap_or_else(|| "None".to_owned());
        format!(
            "query={}, filter={}, limit={:?}",
            structured_query.query(),
            filter,
            structured_query.limit()
        )
    }
}

#[test]
fn core_env_reports_runtime_metadata() {
    let env = get_runtime_environment();

    assert_eq!(env.library, "langchain-core");
    assert_eq!(env.runtime, "rust");
    assert_eq!(env.library_version, langchain_core::VERSION);
}

#[test]
fn chat_loader_collects_sessions() {
    let loader = StaticChatLoader;
    let sessions = loader.load();

    assert_eq!(sessions.len(), 2);
    assert_eq!(sessions[0].messages()[0].content(), "hello");
    assert_eq!(sessions[0].functions()[0]["name"], "lookup");
    assert_eq!(sessions[1].messages()[0].content(), "world");
}

#[tokio::test]
async fn stores_round_trip_sync_and_async_apis() {
    let store = InMemoryStore::<Value>::new();
    store.mset(vec![
        ("alpha".to_owned(), json!("A")),
        ("beta".to_owned(), json!("B")),
    ]);

    assert_eq!(
        store.mget(&["alpha".to_owned(), "beta".to_owned()])[0],
        Some(json!("A"))
    );
    assert_eq!(
        store.yield_keys(Some("a")).collect::<Vec<_>>(),
        vec!["alpha".to_owned()]
    );

    store.amset(vec![("gamma".to_owned(), json!("C"))]).await;
    let async_values = store
        .amget(vec!["gamma".to_owned(), "missing".to_owned()])
        .await;
    assert_eq!(async_values[0], Some(json!("C")));
    assert_eq!(async_values[1], None);

    store.mdelete(&["beta".to_owned()]);
    assert_eq!(store.mget(&["beta".to_owned()])[0], None);

    let byte_store = InMemoryByteStore::new();
    byte_store
        .amset(vec![("blob".to_owned(), vec![1_u8, 2_u8, 3_u8])])
        .await;
    assert_eq!(
        byte_store.amget(vec!["blob".to_owned()]).await[0],
        Some(vec![1, 2, 3])
    );
}

#[test]
fn structured_query_supports_visitor_pattern() {
    let query = StructuredQuery::new("weather")
        .with_filter(FilterDirective::Operation(Operation::new(
            Operator::And,
            vec![
                FilterDirective::Comparison(Comparison::new(
                    Comparator::Eq,
                    "city",
                    json!("Shanghai"),
                )),
                FilterDirective::Comparison(Comparison::new(
                    Comparator::Gte,
                    "temperature",
                    json!(20),
                )),
            ],
        )))
        .with_limit(5);

    let rendered = query.accept(&mut QueryPrinter);

    assert!(rendered.contains("query=weather"));
    assert!(rendered.contains("And("));
    assert!(rendered.contains("Eq(city, \"Shanghai\")"));
    assert!(rendered.contains("Gte(temperature, 20)"));
    assert!(rendered.contains("limit=Some(5)"));
}

#[test]
fn core_agent_schema_reconstructs_messages() {
    let action = AgentAction::new("lookup", json!({"query": "rust"}), "thinking");
    let action_messages = convert_agent_action_to_messages(&action.clone().into());
    assert_eq!(action_messages.len(), 1);
    assert!(matches!(action_messages[0], BaseMessage::Ai(_)));

    let message_log = AgentActionMessageLog::new(
        action.clone(),
        vec![BaseMessage::from(HumanMessage::new("original question"))],
    );
    let logged_messages = convert_agent_action_to_messages(&message_log.into());
    assert_eq!(logged_messages[0].content(), "original question");

    let observation_messages =
        convert_agent_observation_to_messages(&action.into(), &json!({"answer": "done"}));
    assert_eq!(observation_messages.len(), 1);
    assert!(observation_messages[0].content().contains("done"));

    let finish = AgentFinish::new(
        BTreeMap::from([("output".to_owned(), json!("ok"))]),
        "final",
    );
    let step = AgentStep::new(
        AgentAction::new("lookup", json!("input"), "log"),
        json!("observed"),
    );
    assert_eq!(finish.messages()[0].content(), "final");
    assert_eq!(step.messages()[0].content(), "observed");
}

#[test]
fn sys_info_report_mentions_core_version() {
    let report = system_info_report(&["serde"]);

    assert!(report.contains("langchain-core"));
    assert!(report.contains(langchain_core::VERSION));
    assert!(report.contains("serde"));
}
