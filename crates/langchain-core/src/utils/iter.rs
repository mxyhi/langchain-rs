pub fn batch_iterate<T>(
    batch_size: Option<usize>,
    values: impl IntoIterator<Item = T>,
) -> impl Iterator<Item = Vec<T>> {
    let batch_size = batch_size.unwrap_or(usize::MAX).max(1);
    let mut iter = values.into_iter();

    std::iter::from_fn(move || {
        let mut batch = Vec::new();
        for _ in 0..batch_size {
            match iter.next() {
                Some(value) => batch.push(value),
                None => break,
            }
        }

        (!batch.is_empty()).then_some(batch)
    })
}
