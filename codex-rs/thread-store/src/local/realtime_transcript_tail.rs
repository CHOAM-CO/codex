use std::io::BufRead;
use std::io::BufReader;
use std::io::Read;
use std::path::Path;

use codex_protocol::ThreadId;
use codex_protocol::protocol::HistoryPosition;
use codex_protocol::protocol::ThreadHistoryMode;
use codex_rollout::ResponseItemEnvelope;
use codex_rollout::RolloutItem;

use super::LocalThreadStore;
use super::thread_rollout_resolver;
use crate::LoadThreadHistoryParams;
use crate::ThreadStoreError;
use crate::ThreadStoreResult;
use crate::store::is_local_realtime_transcript_tail;

pub(super) async fn load(
    store: &LocalThreadStore,
    params: LoadThreadHistoryParams,
) -> ThreadStoreResult<Vec<ResponseItemEnvelope>> {
    let resolved = if params.include_archived {
        thread_rollout_resolver::resolve_current_including_archived(store, params.thread_id).await?
    } else {
        thread_rollout_resolver::resolve_current(store, params.thread_id).await?
    }
    .ok_or(ThreadStoreError::ThreadNotFound {
        thread_id: params.thread_id,
    })?;
    let path = resolved.path;
    let session_meta = codex_rollout::read_session_meta_line(&path)
        .await
        .map_err(|err| read_error(&path, err))?;
    if session_meta.meta.id != params.thread_id {
        return Err(ThreadStoreError::InvalidRequest {
            message: format!(
                "rollout at {} belongs to thread {}, not {}",
                path.display(),
                session_meta.meta.id,
                params.thread_id
            ),
        });
    }

    let ranges = if session_meta.meta.history_mode == ThreadHistoryMode::Paginated {
        store
            .resolve_rollout_lineage(params.thread_id, Some(path))
            .await?
            .segments()
            .iter()
            // A revert can retain a prefix from an older rollout of this same logical thread.
            // A child must not receive its parent's transcript tails through a history reference.
            .filter(|segment| segment.thread_id == params.thread_id)
            .map(|segment| {
                (
                    segment.rollout_path.clone(),
                    Some(segment.start_ordinal()),
                    segment.end,
                )
            })
            .collect::<Vec<_>>()
    } else {
        vec![(path, None, None)]
    };
    tokio::task::spawn_blocking(move || {
        let mut tails = Vec::new();
        for (path, start_ordinal, end) in ranges {
            read_tails(&path, params.thread_id, start_ordinal, end, &mut tails)?;
        }
        Ok(tails)
    })
    .await
    .map_err(|err| ThreadStoreError::Internal {
        message: format!("failed to join local transcript tail read: {err}"),
    })?
}

fn read_tails(
    path: &Path,
    thread_id: ThreadId,
    start_ordinal: Option<u64>,
    end: Option<HistoryPosition>,
    tails: &mut Vec<ResponseItemEnvelope>,
) -> ThreadStoreResult<()> {
    let file =
        codex_rollout::open_rollout_seekable_reader(path).map_err(|err| read_error(path, err))?;
    // Frozen bounds address the original JSONL bytes even when the file is compressed.
    let readable_bytes = match end {
        Some(end) => end.end_byte_offset,
        None => file.metadata().map_err(|err| read_error(path, err))?.len(),
    };
    let mut reader = BufReader::new(file.take(readable_bytes));
    let mut bytes = Vec::new();
    loop {
        bytes.clear();
        if reader
            .read_until(b'\n', &mut bytes)
            .map_err(|err| read_error(path, err))?
            == 0
        {
            return Ok(());
        }
        // A live writer can append after the caller's flush. As in the existing projection
        // reader, leave an incomplete final record for the next read of the durable prefix.
        if end.is_none() && bytes.last() != Some(&b'\n') {
            return Ok(());
        }
        if bytes.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let line =
            codex_rollout::parse_rollout_line_bytes(&bytes).map_err(|err| read_error(path, err))?;
        if let Some(start) = start_ordinal {
            let ordinal = line.ordinal.ok_or_else(|| ThreadStoreError::Internal {
                message: format!(
                    "paginated rollout {} has a record without an ordinal",
                    path.display()
                ),
            })?;
            if ordinal < start || end.is_some_and(|end| ordinal >= end.end_ordinal_exclusive) {
                continue;
            }
        }
        if let RolloutItem::ResponseItem(envelope) = line.item
            && is_local_realtime_transcript_tail(&envelope, thread_id)
        {
            tails.push(envelope);
        }
    }
}

fn read_error(path: &Path, error: impl std::fmt::Display) -> ThreadStoreError {
    ThreadStoreError::Internal {
        message: format!(
            "failed to read local transcript tails {}: {error}",
            path.display()
        ),
    }
}
