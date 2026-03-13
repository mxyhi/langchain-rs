pub use uuid::Uuid;

pub fn uuid7() -> Uuid {
    Uuid::now_v7()
}
