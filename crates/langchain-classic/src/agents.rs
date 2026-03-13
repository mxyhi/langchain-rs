use std::ops::Deref;

use langchain_core::language_models::BaseChatModel;

pub use langchain::agents::*;

#[derive(Clone)]
struct LegacyAgentChain {
    agent: Agent,
    strategy_name: &'static str,
}

impl LegacyAgentChain {
    fn new(model: impl BaseChatModel + 'static, strategy_name: &'static str) -> Self {
        Self {
            agent: Agent::new(model),
            strategy_name,
        }
    }

    const fn strategy_name(&self) -> &'static str {
        self.strategy_name
    }
}

macro_rules! define_legacy_agent_chain {
    ($name:ident, $strategy_name:literal) => {
        #[derive(Clone)]
        pub struct $name {
            inner: LegacyAgentChain,
        }

        impl $name {
            pub fn new(model: impl BaseChatModel + 'static) -> Self {
                Self {
                    inner: LegacyAgentChain::new(model, $strategy_name),
                }
            }

            pub const fn strategy_name(&self) -> &'static str {
                self.inner.strategy_name()
            }
        }

        impl Deref for $name {
            type Target = Agent;

            fn deref(&self) -> &Self::Target {
                &self.inner.agent
            }
        }
    };
}

define_legacy_agent_chain!(MRKLChain, "mrkl");
define_legacy_agent_chain!(ReActChain, "react");
define_legacy_agent_chain!(SelfAskWithSearchChain, "self-ask-with-search");
