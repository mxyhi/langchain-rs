use futures_util::future::BoxFuture;
use langchain_core::LangChainError;
use langchain_core::language_models::{BaseChatModel, BaseLLM};
use langchain_core::messages::{BaseMessage, HumanMessage};
use langchain_core::prompts::{
    ChatPromptTemplate, PromptArgument, PromptArguments, PromptMessageTemplate, PromptTemplate,
};
use langchain_core::runnables::{Runnable, RunnableConfig};

pub use crate::example_generator::generate_example;

pub struct LLMChain {
    llm: Box<dyn BaseLLM>,
    prompt: PromptTemplate,
}

impl LLMChain {
    pub fn new(llm: impl BaseLLM + 'static, prompt: PromptTemplate) -> Self {
        Self {
            llm: Box::new(llm),
            prompt,
        }
    }

    pub async fn run(&self, arguments: PromptArguments) -> Result<String, LangChainError> {
        let prompt = self.prompt.format(&arguments)?;
        self.llm.invoke(prompt, RunnableConfig::default()).await
    }
}

impl Runnable<PromptArguments, String> for LLMChain {
    fn invoke<'a>(
        &'a self,
        input: PromptArguments,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<String, LangChainError>> {
        Box::pin(async move { self.run(input).await })
    }
}

pub struct ConversationChain {
    model: Box<dyn BaseChatModel>,
    prompt: ChatPromptTemplate,
}

impl ConversationChain {
    pub fn new(model: impl BaseChatModel + 'static) -> Self {
        Self {
            model: Box::new(model),
            prompt: ChatPromptTemplate::from_messages([
                PromptMessageTemplate::placeholder("history"),
                PromptMessageTemplate::human("{input}"),
            ]),
        }
    }

    pub fn with_system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.prompt = ChatPromptTemplate::from_messages([
            PromptMessageTemplate::system(system_prompt),
            PromptMessageTemplate::placeholder("history"),
            PromptMessageTemplate::human("{input}"),
        ]);
        self
    }

    pub async fn predict(
        &self,
        input: impl Into<String>,
        history: Vec<BaseMessage>,
    ) -> Result<String, LangChainError> {
        let arguments = PromptArguments::from([
            ("history".to_owned(), PromptArgument::Messages(history)),
            ("input".to_owned(), PromptArgument::String(input.into())),
        ]);
        let messages = self.prompt.format_messages(&arguments)?;
        let response = self
            .model
            .generate(messages, RunnableConfig::default())
            .await?;

        Ok(response.content().to_owned())
    }

    pub fn seed_history(input: impl Into<String>) -> Vec<BaseMessage> {
        vec![HumanMessage::new(input).into()]
    }
}
