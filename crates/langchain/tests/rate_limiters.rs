use std::time::Duration;

use langchain::rate_limiters::{BaseRateLimiter, InMemoryRateLimiter};

#[tokio::test]
async fn facade_reexports_rate_limiters() {
    let limiter = InMemoryRateLimiter::new(2, Duration::from_millis(5));

    limiter
        .acquire()
        .await
        .expect("first acquire should succeed");
    limiter
        .acquire()
        .await
        .expect("second acquire should succeed");
}
