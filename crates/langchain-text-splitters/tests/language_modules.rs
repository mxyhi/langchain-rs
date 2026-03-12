use langchain_text_splitters::{
    ElementType, ExperimentalMarkdownSyntaxTextSplitter, HTMLHeaderTextSplitter,
    HTMLSectionSplitter, HTMLSemanticPreservingSplitter, HeaderType, HtmlTextSplitter,
    JSFrameworkTextSplitter, JsonTextSplitter, JsxTextSplitter, KonlpyTextSplitter, Language,
    LatexTextSplitter, LineType, MarkdownHeaderTextSplitter, MarkdownTextSplitter,
    NLTKTextSplitter, PythonCodeTextSplitter, RecursiveCharacterTextSplitter,
    RecursiveJsonSplitter, SentenceTransformersTokenTextSplitter, SpacyTextSplitter, TextSplitter,
};

#[test]
fn reference_facade_exports_compile() {
    let exported_types = [
        std::any::type_name::<ElementType>(),
        std::any::type_name::<HeaderType>(),
        std::any::type_name::<LineType>(),
    ];

    assert_eq!(exported_types.len(), 3);
    assert!(std::any::type_name::<HtmlTextSplitter>().contains("HTML"));
    assert!(std::any::type_name::<JsonTextSplitter>().contains("RecursiveJsonSplitter"));
    assert!(std::any::type_name::<JsxTextSplitter>().contains("JSFrameworkTextSplitter"));
}

#[test]
fn recursive_character_text_splitter_can_use_language_presets() {
    let splitter = RecursiveCharacterTextSplitter::from_language(Language::Markdown, 28, 0);
    let text = "# Title\n\n## Section\nalpha beta gamma\n\n## Next\nomega";
    let chunks = splitter.split_text(text);

    assert!(chunks.len() >= 2);
    assert!(chunks.iter().any(|chunk| chunk.contains("## Section")));
    assert!(chunks.iter().any(|chunk| chunk.contains("## Next")));
}

#[test]
fn markdown_and_html_header_splitters_emit_structured_documents() {
    let markdown = MarkdownHeaderTextSplitter::new(
        vec![
            ("#".to_owned(), "title".to_owned()),
            ("##".to_owned(), "section".to_owned()),
        ],
        false,
        false,
    );
    let markdown_docs = markdown.split_text("# Intro\nbody\n## Details\nmore");
    assert_eq!(markdown_docs.len(), 2);
    assert_eq!(markdown_docs[0].metadata["title"].as_str(), Some("Intro"));
    assert_eq!(
        markdown_docs[1].metadata["section"].as_str(),
        Some("Details")
    );

    let html = HTMLHeaderTextSplitter::new(
        vec![
            ("h1".to_owned(), "title".to_owned()),
            ("h2".to_owned(), "section".to_owned()),
        ],
        false,
    );
    let html_docs = html.split_text("<h1>Intro</h1><p>body</p><h2>Details</h2><p>more</p>");
    assert_eq!(html_docs.len(), 2);
    assert_eq!(html_docs[0].metadata["title"].as_str(), Some("Intro"));
    assert_eq!(html_docs[1].metadata["section"].as_str(), Some("Details"));
}

#[test]
fn markdown_and_python_splitters_preserve_structural_boundaries() {
    let markdown = MarkdownTextSplitter::new(26, 0);
    let markdown_chunks = markdown.split_text("# Intro\n\n## One\nalpha beta\n\n## Two\nomega");
    assert!(markdown_chunks.iter().any(|chunk| chunk.contains("## One")));
    assert!(markdown_chunks.iter().any(|chunk| chunk.contains("## Two")));

    let experimental = ExperimentalMarkdownSyntaxTextSplitter::new(false, false);
    let experimental_docs = experimental.split_text("# Intro\n\n```rust\nfn main() {}\n```\n");
    assert_eq!(experimental_docs.len(), 1);
    assert!(experimental_docs[0].page_content.contains("fn main"));

    let python = PythonCodeTextSplitter::new(24, 0);
    let python_chunks = python.split_text(
        "class Cat:\n    pass\n\ndef greet():\n    return 'hi'\n\ndef nap():\n    return True\n",
    );
    assert!(
        python_chunks
            .iter()
            .any(|chunk| chunk.contains("class Cat"))
    );
    assert!(
        python_chunks
            .iter()
            .any(|chunk| chunk.contains("def greet"))
    );
}

#[test]
fn html_json_jsx_and_latex_splitters_expose_real_boundaries() {
    let html = HtmlTextSplitter::new(24, 0);
    let html_chunks =
        html.split_text("<article><h1>Title</h1><p>alpha beta</p><p>omega</p></article>");
    assert!(
        html_chunks
            .iter()
            .any(|chunk| chunk.contains("<h1>Title</h1>"))
    );

    let section = HTMLSectionSplitter::new(24, 0);
    let section_chunks = section.split_text("<section><h2>Title</h2><p>alpha</p></section>");
    assert!(
        section_chunks
            .iter()
            .any(|chunk| chunk.contains("<section>"))
    );

    let semantic = HTMLSemanticPreservingSplitter::new(24, 0);
    let semantic_chunks = semantic.split_text("<main><h1>Title</h1><p>alpha beta</p></main>");
    assert!(semantic_chunks.iter().any(|chunk| chunk.contains("<main>")));

    let json = JsonTextSplitter::new(40, 20);
    let json_chunks = json.split_text("{\"alpha\":1,\"beta\":{\"nested\":true},\"gamma\":3}");
    assert!(json_chunks.iter().any(|chunk| chunk.contains("\"beta\"")));

    let recursive_json = RecursiveJsonSplitter::new(40, 20);
    let recursive_json_chunks =
        recursive_json.split_text("{\"team\":{\"name\":\"langchain\",\"size\":3}}");
    assert!(
        recursive_json_chunks
            .iter()
            .any(|chunk| chunk.contains("\"team\""))
    );

    let jsx = JsxTextSplitter::new(36, 0);
    let jsx_chunks = jsx.split_text(
        "export function Card() {\n  return <section><h1>Title</h1></section>;\n}\nconst footer = <footer>Done</footer>;\n",
    );
    assert!(
        jsx_chunks
            .iter()
            .any(|chunk| chunk.contains("export function Card"))
    );

    let framework = JSFrameworkTextSplitter::new(36, 0);
    let framework_chunks =
        framework.split_text("const App = () => <main><Widget /></main>;\nexport default App;\n");
    assert!(
        framework_chunks
            .iter()
            .any(|chunk| chunk.contains("export default"))
    );

    let latex = LatexTextSplitter::new(24, 0);
    let latex_chunks = latex.split_text(
        "\\section{Intro}\nalpha beta\n\\subsection{Details}\nomega\n\\begin{itemize}\n\\item one\n\\end{itemize}\n",
    );
    assert!(
        latex_chunks
            .iter()
            .any(|chunk| chunk.contains("\\section{Intro}"))
    );
}

#[test]
fn compatibility_sentence_splitters_handle_natural_language() {
    let nltk = NLTKTextSplitter::new(18, 0);
    let nltk_chunks = nltk.split_text("Alpha is here. Beta follows! Gamma stays?");
    assert_eq!(
        nltk_chunks,
        vec!["Alpha is here.", "Beta follows!", "Gamma stays?"]
    );

    let spacy = SpacyTextSplitter::new(24, 0);
    let spacy_chunks = spacy.split_text("Alpha is here.\nBeta follows without pause. Gamma stays.");
    assert_eq!(spacy_chunks.len(), 3);
    assert!(spacy_chunks[1].starts_with("Beta"));

    let konlpy = KonlpyTextSplitter::new(16, 0);
    let konlpy_chunks = konlpy.split_text("첫 문장입니다. 둘째 문장입니다! 셋째 문장입니다?");
    assert_eq!(konlpy_chunks.len(), 3);
}

#[test]
fn sentence_transformers_token_splitter_uses_token_windows() {
    let splitter = SentenceTransformersTokenTextSplitter::new(3, 1);
    let chunks = splitter.split_text("alpha beta gamma delta epsilon");

    assert_eq!(chunks, vec!["alpha beta gamma", "gamma delta epsilon"]);
}
