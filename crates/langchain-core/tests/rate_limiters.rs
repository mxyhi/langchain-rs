use std::time::{Duration, Instant};

use langchain_core::rate_limiters::{BaseRateLimiter, InMemoryRateLimiter};

#[tokio::test]
async fn in_memory_rate_limiter_allows_burst_then_waits() {
    let limiter = InMemoryRateLimiter::new(1, Duration::from_millis(20));

    limiter
        .acquire()
        .await
        .expect("first acquire should succeed");
    let start = Instant::now();
    limiter
        .acquire()
        .await
        .expect("second acquire should wait then succeed");

    assert!(
        start.elapsed() >= Duration::from_millis(15),
        "second acquire should respect the interval"
    );
}

#[tokio::test]
async fn in_memory_rate_limiter_keeps_configuration() {
    let limiter = InMemoryRateLimiter::new(3, Duration::from_millis(25));

    assert_eq!(limiter.max_calls(), 3);
    assert_eq!(limiter.interval(), Duration::from_millis(25));
}
