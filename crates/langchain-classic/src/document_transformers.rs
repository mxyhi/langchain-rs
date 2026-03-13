use langchain_core::documents::Document;

pub const CLASSIC_PACKAGE: &str = crate::_api::CLASSIC_PACKAGE;

pub trait DocumentTransformer {
    fn transform_documents(&self, documents: Vec<Document>) -> Vec<Document>;
}

fn strip_html_tags(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut in_tag = false;

    for character in input.chars() {
        match character {
            '<' => in_tag = true,
            '>' => {
                in_tag = false;
                output.push(' ');
            }
            _ if !in_tag => output.push(character),
            _ => {}
        }
    }

    output.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Html2TextTransformer;

impl Html2TextTransformer {
    pub const fn new() -> Self {
        Self
    }

    pub fn transform_documents(&self, documents: Vec<Document>) -> Vec<Document> {
        <Self as DocumentTransformer>::transform_documents(self, documents)
    }
}

impl DocumentTransformer for Html2TextTransformer {
    fn transform_documents(&self, documents: Vec<Document>) -> Vec<Document> {
        documents
            .into_iter()
            .map(|mut document| {
                document.page_content = strip_html_tags(&document.page_content);
                document
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BeautifulSoupTransformer;

impl BeautifulSoupTransformer {
    pub const fn new() -> Self {
        Self
    }

    pub fn transform_documents(&self, documents: Vec<Document>) -> Vec<Document> {
        <Self as DocumentTransformer>::transform_documents(self, documents)
    }
}

impl DocumentTransformer for BeautifulSoupTransformer {
    fn transform_documents(&self, documents: Vec<Document>) -> Vec<Document> {
        Html2TextTransformer::new().transform_documents(documents)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LongContextReorder;

impl LongContextReorder {
    pub const fn new() -> Self {
        Self
    }

    pub fn transform_documents(&self, documents: Vec<Document>) -> Vec<Document> {
        <Self as DocumentTransformer>::transform_documents(self, documents)
    }
}

impl DocumentTransformer for LongContextReorder {
    fn transform_documents(&self, documents: Vec<Document>) -> Vec<Document> {
        let mut ordered = Vec::with_capacity(documents.len());
        let mut front = 0_usize;
        let mut back = documents.len();
        let mut take_front = true;

        while front < back {
            if take_front {
                ordered.push(documents[front].clone());
                front += 1;
            } else {
                back -= 1;
                ordered.push(documents[back].clone());
            }
            take_front = !take_front;
        }

        ordered
    }
}

macro_rules! define_passthrough_transformer {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, Default)]
        pub struct $name;

        impl $name {
            pub const fn new() -> Self {
                Self
            }

            pub fn transform_documents(&self, documents: Vec<Document>) -> Vec<Document> {
                <Self as DocumentTransformer>::transform_documents(self, documents)
            }
        }

        impl DocumentTransformer for $name {
            fn transform_documents(&self, documents: Vec<Document>) -> Vec<Document> {
                documents
            }
        }
    };
}

define_passthrough_transformer!(DoctranPropertyExtractor);
define_passthrough_transformer!(DoctranQATransformer);
define_passthrough_transformer!(DoctranTextTranslator);
define_passthrough_transformer!(EmbeddingsClusteringFilter);
define_passthrough_transformer!(EmbeddingsRedundantFilter);
define_passthrough_transformer!(GoogleTranslateTransformer);
define_passthrough_transformer!(NucliaTextTransformer);
define_passthrough_transformer!(OpenAIMetadataTagger);

pub fn get_stateful_documents(documents: Vec<Document>) -> Vec<Document> {
    documents
}
