use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::sync::Arc;

use futures_util::future::{BoxFuture, try_join_all};
use futures_util::stream::{self, BoxStream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::LangChainError;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RunnableConfig {
    pub tags: Vec<String>,
    pub metadata: BTreeMap<String, Value>,
    pub configurable: BTreeMap<String, Value>,
    pub max_concurrency: Option<usize>,
}

pub trait Runnable<I, O>: Send + Sync
where
    I: Send + 'static,
    O: Send + 'static,
{
    fn invoke<'a>(
        &'a self,
        input: I,
        config: RunnableConfig,
    ) -> BoxFuture<'a, Result<O, LangChainError>>;

    fn batch<'a>(
        &'a self,
        inputs: Vec<I>,
        config: RunnableConfig,
    ) -> BoxFuture<'a, Result<Vec<O>, LangChainError>> {
        Box::pin(async move {
            try_join_all(
                inputs
                    .into_iter()
                    .map(|input| self.invoke(input, config.clone())),
            )
            .await
        })
    }

    fn stream<'a>(
        &'a self,
        input: I,
        config: RunnableConfig,
    ) -> BoxStream<'a, Result<O, LangChainError>> {
        stream::once(self.invoke(input, config)).boxed()
    }

    fn pipe<Next, NextOutput>(self, next: Next) -> RunnableSequence<Self, Next, O>
    where
        Self: Sized,
        NextOutput: Send + 'static,
        Next: Runnable<O, NextOutput>,
    {
        RunnableSequence::new(self, next)
    }
}

pub trait RunnableDyn<I, O>: Send + Sync {
    fn invoke_boxed<'a>(
        &'a self,
        input: I,
        config: RunnableConfig,
    ) -> BoxFuture<'a, Result<O, LangChainError>>;
}

impl<T, I, O> RunnableDyn<I, O> for T
where
    T: Runnable<I, O>,
    I: Send + 'static,
    O: Send + 'static,
{
    fn invoke_boxed<'a>(
        &'a self,
        input: I,
        config: RunnableConfig,
    ) -> BoxFuture<'a, Result<O, LangChainError>> {
        Box::pin(self.invoke(input, config))
    }
}

pub struct RunnableLambda<F> {
    function: F,
}

impl<F> RunnableLambda<F> {
    pub fn new(function: F) -> Self {
        Self { function }
    }
}

impl<F, Fut, I, O> Runnable<I, O> for RunnableLambda<F>
where
    I: Send + 'static,
    O: Send + 'static,
    F: Fn(I) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = Result<O, LangChainError>> + Send + 'static,
{
    fn invoke<'a>(
        &'a self,
        input: I,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<O, LangChainError>> {
        Box::pin((self.function)(input))
    }
}

pub struct RunnableSequence<A, B, Middle> {
    first: A,
    second: B,
    _middle: PhantomData<fn() -> Middle>,
}

impl<A, B, Middle> RunnableSequence<A, B, Middle> {
    pub fn new(first: A, second: B) -> Self {
        Self {
            first,
            second,
            _middle: PhantomData,
        }
    }
}

impl<A, B, I, O, Middle> Runnable<I, O> for RunnableSequence<A, B, Middle>
where
    A: Runnable<I, Middle>,
    B: Runnable<Middle, O>,
    I: Send + 'static,
    Middle: Send + 'static,
    O: Send + 'static,
{
    fn invoke<'a>(
        &'a self,
        input: I,
        config: RunnableConfig,
    ) -> BoxFuture<'a, Result<O, LangChainError>> {
        Box::pin(async move {
            let middle = self.first.invoke(input, config.clone()).await?;
            self.second.invoke(middle, config).await
        })
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RunnablePassthrough;

impl RunnablePassthrough {
    pub fn new() -> Self {
        Self
    }
}

impl<T> Runnable<T, T> for RunnablePassthrough
where
    T: Send + 'static,
{
    fn invoke<'a>(
        &'a self,
        input: T,
        _config: RunnableConfig,
    ) -> BoxFuture<'a, Result<T, LangChainError>> {
        Box::pin(async move { Ok(input) })
    }
}

pub struct RunnableParallel<I, O> {
    branches: BTreeMap<String, Arc<dyn RunnableDyn<I, O>>>,
}

impl<I, O> Default for RunnableParallel<I, O> {
    fn default() -> Self {
        Self {
            branches: BTreeMap::new(),
        }
    }
}

impl<I, O> RunnableParallel<I, O> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_branch(
        mut self,
        name: impl Into<String>,
        branch: impl Runnable<I, O> + 'static,
    ) -> Self
    where
        I: Send + 'static,
        O: Send + 'static,
    {
        self.branches.insert(name.into(), Arc::new(branch));
        self
    }
}

impl<I, O> Runnable<I, BTreeMap<String, O>> for RunnableParallel<I, O>
where
    I: Clone + Send + Sync + 'static,
    O: Send + 'static,
{
    fn invoke<'a>(
        &'a self,
        input: I,
        config: RunnableConfig,
    ) -> BoxFuture<'a, Result<BTreeMap<String, O>, LangChainError>> {
        Box::pin(async move {
            let mut pending = Vec::with_capacity(self.branches.len());

            for (name, branch) in &self.branches {
                pending.push(async {
                    let output = branch.invoke_boxed(input.clone(), config.clone()).await?;
                    Ok::<_, LangChainError>((name.clone(), output))
                });
            }

            Ok(try_join_all(pending).await?.into_iter().collect())
        })
    }
}
