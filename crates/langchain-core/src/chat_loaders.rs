use crate::chat_sessions::ChatSession;

pub trait BaseChatLoader {
    fn lazy_load<'a>(&'a self) -> Box<dyn Iterator<Item = ChatSession> + 'a>;

    fn load(&self) -> Vec<ChatSession> {
        self.lazy_load().collect()
    }
}
