pub(crate) fn group_segments(
    segments: Vec<String>,
    chunk_size: usize,
    chunk_overlap: usize,
    separator: &str,
) -> Vec<String> {
    let segments = segments
        .into_iter()
        .map(|segment| segment.trim().to_owned())
        .filter(|segment| !segment.is_empty())
        .collect::<Vec<_>>();
    if segments.is_empty() {
        return Vec::new();
    }

    let chunk_size = chunk_size.max(1);
    let mut chunks = Vec::new();
    let mut start = 0;

    while start < segments.len() {
        let mut end = start;
        let mut current = String::new();

        while end < segments.len() {
            let candidate = if current.is_empty() {
                segments[end].clone()
            } else {
                format!("{current}{separator}{}", segments[end])
            };

            if !current.is_empty() && candidate.chars().count() > chunk_size {
                break;
            }

            current = candidate;
            end += 1;
        }

        if current.is_empty() {
            current = segments[start].clone();
            end = start + 1;
        }

        chunks.push(current);
        if end >= segments.len() {
            break;
        }

        // Overlap is applied in segment units so sentence-style splitters can
        // reuse prior segments without re-tokenizing text.
        let consumed = end - start;
        let overlap = chunk_overlap.min(consumed.saturating_sub(1));
        start = end - overlap;
    }

    chunks
}

pub(crate) fn split_sentences(text: &str, split_on_newlines: bool) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for character in text.chars() {
        if split_on_newlines && character == '\n' {
            flush_buffer(&mut current, &mut sentences);
            continue;
        }

        current.push(character);
        if matches!(character, '.' | '!' | '?' | '。' | '！' | '？') {
            flush_buffer(&mut current, &mut sentences);
        }
    }

    flush_buffer(&mut current, &mut sentences);
    sentences
}

fn flush_buffer(buffer: &mut String, output: &mut Vec<String>) {
    let normalized = buffer.trim();
    if !normalized.is_empty() {
        output.push(normalized.to_owned());
    }
    buffer.clear();
}
