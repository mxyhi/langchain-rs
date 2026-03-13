use futures_util::Stream;
use futures_util::StreamExt;

pub fn abatch_iterate<T, S>(batch_size: Option<usize>, stream: S) -> impl Stream<Item = Vec<T>>
where
    S: Stream<Item = T> + Unpin,
{
    let batch_size = batch_size.unwrap_or(usize::MAX).max(1);

    futures_util::stream::unfold(stream, move |mut stream| async move {
        let mut batch = Vec::new();

        while batch.len() < batch_size {
            match stream.next().await {
                Some(value) => batch.push(value),
                None => break,
            }
        }

        (!batch.is_empty()).then_some((batch, stream))
    })
}
