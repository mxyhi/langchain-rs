use std::sync::LazyLock;

use langchain_core::prompts::PromptTemplate;

pub static ENTITY_MEMORY_CONVERSATION_TEMPLATE: LazyLock<PromptTemplate> = LazyLock::new(|| {
    PromptTemplate::new(
        "You are an assistant to a human, powered by a large language model trained by OpenAI.\n\n\
You are designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, you are able to generate human-like text based on the input you receive, allowing you to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.\n\n\
You are constantly learning and improving, and your capabilities are constantly evolving. You are able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. You have access to some personalized information provided by the human in the Context section below. Additionally, you are able to generate your own text based on the input you receive, allowing you to engage in discussions and provide explanations and descriptions on a wide range of topics.\n\n\
Overall, you are a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether the human needs help with a specific question or just wants to have a conversation about a particular topic, you are here to assist.\n\n\
Context:\n\
{entities}\n\n\
Current conversation:\n\
{history}\n\
Last line:\n\
Human: {input}\n\
You:",
    )
});

pub static SUMMARY_PROMPT: LazyLock<PromptTemplate> = LazyLock::new(|| {
    PromptTemplate::new(
        "Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary.\n\n\
EXAMPLE\n\
Current summary:\n\
The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good.\n\n\
New lines of conversation:\n\
Human: Why do you think artificial intelligence is a force for good?\n\
AI: Because artificial intelligence will help humans reach their full potential.\n\n\
New summary:\n\
The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good because it will help humans reach their full potential.\n\
END OF EXAMPLE\n\n\
Current summary:\n\
{summary}\n\n\
New lines of conversation:\n\
{new_lines}\n\n\
New summary:",
    )
});

pub static ENTITY_EXTRACTION_PROMPT: LazyLock<PromptTemplate> = LazyLock::new(|| {
    PromptTemplate::new(
        "You are an AI assistant reading the transcript of a conversation between an AI and a human. Extract all of the proper nouns from the last line of conversation. As a guideline, a proper noun is generally capitalized. You should definitely extract all names and places.\n\n\
The conversation history is provided just in case of a coreference -- ignore items mentioned there that are not in the last line.\n\n\
Return the output as a single comma-separated list, or NONE if there is nothing of note to return.\n\n\
Conversation history (for reference only):\n\
{history}\n\
Last line of conversation (for extraction):\n\
Human: {input}\n\n\
Output:",
    )
});

pub static ENTITY_SUMMARIZATION_PROMPT: LazyLock<PromptTemplate> = LazyLock::new(|| {
    PromptTemplate::new(
        "You are an AI assistant helping a human keep track of facts about relevant people, places, and concepts in their life. Update the summary of the provided entity in the Entity section based on the last line of your conversation with the human.\n\n\
Full conversation history (for context):\n\
{history}\n\n\
Entity to summarize:\n\
{entity}\n\n\
Existing summary of {entity}:\n\
{summary}\n\n\
Last line of conversation:\n\
Human: {input}\n\
Updated summary:",
    )
});

pub const KG_TRIPLE_DELIMITER: &str = "<|>";

pub static KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT: LazyLock<PromptTemplate> = LazyLock::new(|| {
    PromptTemplate::new(format!(
        "You are a networked intelligence helping a human track knowledge triples about relevant entities. Extract all knowledge triples from the last line of conversation.\n\n\
Conversation history (for reference only):\n\
{{history}}\n\n\
Last line of conversation (for extraction):\n\
Human: {{input}}\n\n\
Output triples separated by {KG_TRIPLE_DELIMITER}:"
    ))
});
