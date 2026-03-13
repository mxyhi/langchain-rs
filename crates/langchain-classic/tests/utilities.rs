use langchain_classic::PromptTemplate;
use langchain_classic::hub;
use langchain_classic::prompts::PromptArgument;
use langchain_classic::python::PythonREPL;
use langchain_classic::requests::{Requests, RequestsWrapper};
use langchain_classic::serpapi::SerpAPIWrapper;
use langchain_classic::sql_database::SQLDatabase;
use serde_json::json;
use tempfile::tempdir;
use wiremock::matchers::{body_json, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn requests_wrappers_fetch_text_and_json() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/hello"))
        .respond_with(ResponseTemplate::new(200).set_body_string("world"))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/json"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"ok": true})))
        .mount(&server)
        .await;

    let wrapper = RequestsWrapper::new().with_requests(Requests::new());
    let text = wrapper
        .get_text(&format!("{}/hello", server.uri()))
        .await
        .expect("text request should succeed");
    let value = wrapper
        .get_json(&format!("{}/json", server.uri()))
        .await
        .expect("json request should succeed");

    assert_eq!(text, "world");
    assert_eq!(value, json!({"ok": true}));
}

#[tokio::test]
async fn serpapi_wrapper_builds_expected_query() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/search"))
        .and(query_param("q", "rust langchain"))
        .and(query_param("api_key", "test-key"))
        .and(query_param("engine", "google"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"organic_results": []})))
        .mount(&server)
        .await;

    let wrapper = SerpAPIWrapper::new("test-key").with_endpoint(format!("{}/search", server.uri()));
    let payload = wrapper
        .search("rust langchain")
        .await
        .expect("serpapi search should succeed");

    assert_eq!(payload, json!({"organic_results": []}));
}

#[tokio::test]
async fn python_repl_executes_code() {
    let repl = PythonREPL::new();
    let output = repl.run("print('hello from python')").await;

    match output {
        Ok(stdout) => assert_eq!(stdout.trim(), "hello from python"),
        Err(error) => {
            assert!(
                error.to_string().contains("No such file")
                    || error.to_string().contains("not found"),
                "unexpected python failure: {error}"
            );
        }
    }
}

#[test]
fn sql_database_executes_queries_against_sqlite() {
    let directory = tempdir().expect("temp dir should exist");
    let path = directory.path().join("langchain-classic.sqlite3");
    let database = SQLDatabase::from_sqlite_path(&path);

    database
        .execute_batch(
            "CREATE TABLE notes (id INTEGER PRIMARY KEY, title TEXT, score REAL);
             INSERT INTO notes (title, score) VALUES ('alpha', 1.5), ('beta', 2.0);",
        )
        .expect("sqlite setup should succeed");

    let rows = database
        .query("SELECT title, score FROM notes ORDER BY id")
        .expect("sqlite query should succeed");

    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["title"], json!("alpha"));
    assert_eq!(rows[1]["score"], json!(2.0));
    assert_eq!(
        database.get_usable_table_names().expect("table list"),
        vec!["notes"]
    );
}

#[tokio::test]
async fn hub_push_posts_prompt_manifest_and_returns_browser_url() {
    let server = MockServer::start().await;
    let prompt = PromptTemplate::new("Hello {name}");
    let options = hub::HubOptions::new().with_api_url(server.uri());

    Mock::given(method("POST"))
        .and(path("/repos/owner/prompt"))
        .and(body_json(json!({
            "manifest": {
                "kind": "prompt_template",
                "template": "Hello {name}"
            }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "url": "https://hub.example/prompts/owner/prompt/commit/abc123"
        })))
        .mount(&server)
        .await;

    let url = hub::push_with_options("owner/prompt", &prompt, &options)
        .expect("hub push should succeed against mock server");
    assert_eq!(
        url,
        "https://hub.example/prompts/owner/prompt/commit/abc123"
    );
}

#[tokio::test]
async fn hub_pull_fetches_prompt_template_from_http_manifest() {
    let server = MockServer::start().await;
    let options = hub::HubOptions::new().with_api_url(server.uri());

    Mock::given(method("GET"))
        .and(path("/repos/owner/prompt"))
        .and(query_param("commit", "abc123"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "manifest": {
                "kind": "prompt_template",
                "template": "Hello {name}"
            }
        })))
        .mount(&server)
        .await;

    let prompt = hub::pull_with_options("owner/prompt:abc123", &options)
        .expect("hub pull should succeed against mock server");
    let rendered = prompt
        .format(&[("name".to_owned(), PromptArgument::String("Rust".to_owned()))].into())
        .expect("pulled prompt should render");
    assert_eq!(rendered, "Hello Rust");
}
