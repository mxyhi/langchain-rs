use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures_util::future::BoxFuture;
use tokio::sync::Mutex;
use tokio::time::sleep;

use crate::LangChainError;

pub trait BaseRateLimiter: Send + Sync {
    fn acquire<'a>(&'a self) -> BoxFuture<'a, Result<(), LangChainError>>;
}

#[derive(Debug, Clone)]
pub struct InMemoryRateLimiter {
    max_calls: usize,
    interval: Duration,
    timestamps: Arc<Mutex<VecDeque<Instant>>>,
}

impl InMemoryRateLimiter {
    pub fn new(max_calls: usize, interval: Duration) -> Self {
        Self {
            max_calls: max_calls.max(1),
            interval: interval.max(Duration::from_millis(1)),
            timestamps: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    pub fn max_calls(&self) -> usize {
        self.max_calls
    }

    pub fn interval(&self) -> Duration {
        self.interval
    }
}

impl BaseRateLimiter for InMemoryRateLimiter {
    fn acquire<'a>(&'a self) -> BoxFuture<'a, Result<(), LangChainError>> {
        Box::pin(async move {
            loop {
                let wait_for = {
                    let mut timestamps = self.timestamps.lock().await;
                    let now = Instant::now();

                    while timestamps
                        .front()
                        .is_some_and(|timestamp| now.duration_since(*timestamp) >= self.interval)
                    {
                        timestamps.pop_front();
                    }

                    if timestamps.len() < self.max_calls {
                        timestamps.push_back(now);
                        None
                    } else {
                        timestamps.front().map(|timestamp| {
                            self.interval.saturating_sub(now.duration_since(*timestamp))
                        })
                    }
                };

                match wait_for {
                    Some(duration) if !duration.is_zero() => sleep(duration).await,
                    Some(_) | None => return Ok(()),
                }
            }
        })
    }
}
