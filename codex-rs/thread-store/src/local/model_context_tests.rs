use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;

use codex_protocol::ThreadId;
use codex_protocol::config_types::ReasoningSummary;
use codex_protocol::items::TurnItem;
use codex_protocol::items::UserMessageItem;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::AskForApproval;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::HistoryPosition;
use codex_protocol::protocol::ItemCompletedEvent;
use codex_protocol::protocol::MultiAgentVersion;
use codex_protocol::protocol::SandboxPolicy;
use codex_protocol::protocol::ThreadHistoryMode;
use codex_protocol::protocol::TurnCompleteEvent;
use codex_protocol::protocol::TurnContextItem;
use codex_protocol::protocol::TurnStartedEvent;
use codex_protocol::protocol::WorldStateItem;
use codex_protocol::user_input::UserInput;
use codex_rollout::CompactedItem;
use codex_rollout::RolloutItem;
use codex_rollout::RolloutLine;
use pretty_assertions::assert_eq;
use tempfile::TempDir;
use uuid::Uuid;

use super::*;
use crate::ThreadStore;
use crate::local::test_support::test_config;
use crate::local::test_support::write_session_file_with_history_mode;

#[tokio::test]
async fn stops_at_newest_usable_compaction_and_keeps_companions() {
    let home = TempDir::new().expect("temp dir");
    let uuid = Uuid::from_u128(/*v*/ 1001);
    let thread_id = ThreadId::from_string(&uuid.to_string()).expect("thread id");
    let world_state = RolloutItem::WorldState(WorldStateItem::full(Default::default()));
    let turn_context = turn_context(home.path(), "companion");
    let path = write_paginated_rollout(
        home.path(),
        "2025-01-03T13-00-00",
        uuid,
        [
            user_message("older history"),
            compacted("older compaction", Some(Vec::new())),
            user_message("superseded suffix"),
            compacted("latest compaction", Some(Vec::new())),
            world_state.clone(),
            turn_context.clone(),
        ],
    );
    let session_meta = codex_rollout::read_session_meta_line(path.as_path())
        .await
        .expect("read session metadata");
    let store = LocalThreadStore::new(test_config(home.path()), /*state_db*/ None);

    let context = store
        .load_latest_model_context(LoadThreadHistoryParams {
            thread_id,
            include_archived: false,
        })
        .await
        .expect("load model context");

    assert_eq!(
        serde_json::to_value(context.items).expect("serialize context"),
        serde_json::to_value(vec![
            RolloutItem::SessionMeta(session_meta),
            compacted("latest compaction", Some(Vec::new())),
            world_state,
            turn_context,
        ])
        .expect("serialize expected context")
    );
}

#[tokio::test]
async fn loads_recent_context_after_many_empty_wake_turns() {
    let home = TempDir::new().expect("temp dir");
    let uuid = Uuid::from_u128(/*v*/ 1008);
    let thread_id = ThreadId::from_string(&uuid.to_string()).expect("thread id");
    let path = write_paginated_rollout(
        home.path(),
        "2025-01-03T13-00-07",
        uuid,
        [
            turn_started("user-turn"),
            user_message("keep working"),
            completed_user_message("user-turn", "keep working"),
            turn_context(home.path(), "user-turn"),
            turn_complete("user-turn"),
        ],
    );
    let mut expected_suffix = Vec::new();
    for index in 0..32 {
        let turn_id = format!("wake-{index}");
        let mut items = vec![turn_started(&turn_id)];
        let starts_new_window = index % 8 == 0;
        if starts_new_window {
            items.extend([
                compacted(&format!("compaction-{index}"), Some(Vec::new())),
                RolloutItem::WorldState(WorldStateItem::full(Default::default())),
            ]);
        }
        items.extend([
            turn_context(home.path(), &turn_id),
            contextual_user_message(),
            turn_complete(&turn_id),
        ]);
        append_items(&path, items.clone());
        if starts_new_window {
            expected_suffix = items.into_iter().skip(1).collect();
        } else {
            expected_suffix.extend(items);
        }
    }
    let session_meta = codex_rollout::read_session_meta_line(&path)
        .await
        .expect("read session metadata");
    let store = LocalThreadStore::new(test_config(home.path()), /*state_db*/ None);
    let context = store
        .load_latest_model_context(LoadThreadHistoryParams {
            thread_id,
            include_archived: false,
        })
        .await
        .expect("load model context");
    expected_suffix.insert(0, RolloutItem::SessionMeta(session_meta));

    assert_eq!(
        serde_json::to_value(context.items).expect("serialize context"),
        serde_json::to_value(expected_suffix).expect("serialize expected context")
    );
}

#[tokio::test]
async fn fork_context_excludes_items_after_frozen_cutoff() {
    let home = TempDir::new().expect("temp dir");
    let uuid = Uuid::from_u128(/*v*/ 1007);
    let thread_id = ThreadId::from_string(&uuid.to_string()).expect("thread id");
    let path = write_ordinaled_paginated_rollout(
        home.path(),
        "2025-01-03T13-00-06",
        uuid,
        [turn_started("frozen-turn"), user_message("frozen message")],
    );
    let history_base =
        history_position(path.as_path(), thread_id, /*end_ordinal_exclusive*/ 3);
    append_items(path.as_path(), [user_message("later message")]);
    let store = LocalThreadStore::new(test_config(home.path()), /*state_db*/ None);
    let lineage = store
        .resolve_rollout_lineage(thread_id, /*initial_path*/ None)
        .await
        .expect("resolve source lineage");
    let session_meta = codex_rollout::read_session_meta_line(path.as_path())
        .await
        .expect("read source metadata");

    let context = load_for_fork(lineage, Some(history_base))
        .await
        .expect("load frozen fork context");

    let expected = vec![
        RolloutItem::SessionMeta(session_meta),
        turn_started("frozen-turn"),
        user_message("frozen message"),
    ];
    assert_eq!(
        serde_json::to_value(context).expect("serialize fork context"),
        serde_json::to_value(expected).expect("serialize expected fork context")
    );
}

#[tokio::test]
async fn fork_version_stops_before_older_segments_once_resolved() {
    for stored_version in [None, Some(MultiAgentVersion::Disabled)] {
        let home = TempDir::new().expect("temp dir");
        let root_uuid = Uuid::from_u128(/*v*/ 3001);
        let root_id = ThreadId::from_string(&root_uuid.to_string()).expect("root id");
        let root_path = write_ordinaled_paginated_rollout(
            home.path(),
            "2025-01-03T13-02-00",
            root_uuid,
            [user_message("older history")],
        );
        let child_uuid = Uuid::from_u128(/*v*/ 3002);
        let child_id = ThreadId::from_string(&child_uuid.to_string()).expect("child id");
        let child_path =
            write_ordinaled_paginated_rollout(home.path(), "2025-01-03T13-02-01", child_uuid, []);
        let mut source_meta = codex_rollout::read_session_meta_line(&child_path)
            .await
            .expect("read child metadata");
        source_meta.meta.multi_agent_version = stored_version;
        source_meta.meta.history_base = Some(history_position(
            &root_path, root_id, /*end_ordinal_exclusive*/ 2,
        ));
        let head = RolloutLine {
            timestamp: "2025-01-03T13:02:01Z".to_string(),
            ordinal: Some(0),
            item: RolloutItem::SessionMeta(source_meta.clone()),
        };
        std::fs::write(
            &child_path,
            format!(
                "{}\n",
                serde_json::to_string(&head).expect("serialize head")
            ),
        )
        .expect("write child metadata");
        let RolloutItem::Compacted(mut compacted) =
            compacted("version compaction", Some(Vec::new()))
        else {
            unreachable!();
        };
        compacted.resume_metadata = Some(codex_rollout::CompactionResumeMetadata {
            multi_agent_version: Some(MultiAgentVersion::V2),
            last_started_turn_id: None,
            turn_attribution: None,
            previous_turn_settings: None,
        });
        append_items(
            &child_path,
            [
                RolloutItem::Compacted(compacted),
                turn_context(home.path(), "unset-turn"),
            ],
        );
        let store = LocalThreadStore::new(test_config(home.path()), /*state_db*/ None);
        let lineage = store
            .resolve_rollout_lineage(child_id, /*initial_path*/ None)
            .await
            .expect("resolve source lineage");
        // The usable compaction should stop the scan before it reaches this missing segment.
        std::fs::remove_file(root_path).expect("remove older segment after resolving lineage");

        let context = load_for_fork(lineage, /*history_base*/ None)
            .await
            .expect("resolve version without reading older segments");
        source_meta.meta.multi_agent_version = stored_version.or(Some(MultiAgentVersion::V2));
        assert_eq!(
            serde_json::to_value(context).expect("serialize fork context"),
            serde_json::to_value(vec![RolloutItem::SessionMeta(source_meta)])
                .expect("serialize expected context")
        );
    }
}

#[tokio::test]
async fn fork_version_respects_inherited_segment_cutoffs() {
    let home = TempDir::new().expect("temp dir");
    let root_uuid = Uuid::from_u128(/*v*/ 3003);
    let root_id = ThreadId::from_string(&root_uuid.to_string()).expect("root id");
    let RolloutItem::TurnContext(mut inherited) = turn_context(home.path(), "inherited") else {
        unreachable!();
    };
    inherited.multi_agent_version = Some(MultiAgentVersion::V2);
    let mut excluded = inherited.clone();
    excluded.multi_agent_version = Some(MultiAgentVersion::V1);
    let root_path = write_ordinaled_paginated_rollout(
        home.path(),
        "2025-01-03T13-02-02",
        root_uuid,
        [
            RolloutItem::TurnContext(inherited),
            RolloutItem::TurnContext(excluded),
        ],
    );
    let child_uuid = Uuid::from_u128(/*v*/ 3004);
    let child_id = ThreadId::from_string(&child_uuid.to_string()).expect("child id");
    let child_path = write_ordinaled_paginated_rollout(
        home.path(),
        "2025-01-03T13-02-03",
        child_uuid,
        [turn_context(home.path(), "unset-child")],
    );
    set_history_base(
        &child_path,
        history_position(&root_path, root_id, /*end_ordinal_exclusive*/ 2),
    );
    let store = LocalThreadStore::new(test_config(home.path()), /*state_db*/ None);
    let lineage = store
        .resolve_rollout_lineage(child_id, /*initial_path*/ None)
        .await
        .expect("resolve source lineage");
    let mut source_meta = codex_rollout::read_session_meta_line(&child_path)
        .await
        .expect("read child metadata");

    let context = load_for_fork(lineage, /*history_base*/ None)
        .await
        .expect("recover version from inherited prefix");
    source_meta.meta.multi_agent_version = Some(MultiAgentVersion::V2);
    assert_eq!(
        serde_json::to_value(context).expect("serialize fork context"),
        serde_json::to_value(vec![RolloutItem::SessionMeta(source_meta)])
            .expect("serialize expected context")
    );
}

#[tokio::test]
async fn returns_scanned_full_history_for_unsupported_compaction() {
    enum MissingField {
        ReplacementHistory,
        WindowNumber,
    }

    for (index, missing_field) in [MissingField::ReplacementHistory, MissingField::WindowNumber]
        .into_iter()
        .enumerate()
    {
        let home = TempDir::new().expect("temp dir");
        let uuid = Uuid::from_u128(1002 + index as u128);
        let mut unsupported = compacted("unsupported compaction", Some(Vec::new()));
        let RolloutItem::Compacted(unsupported_compaction) = &mut unsupported else {
            unreachable!("compacted helper returns a compaction");
        };
        match missing_field {
            MissingField::ReplacementHistory => unsupported_compaction.replacement_history = None,
            MissingField::WindowNumber => unsupported_compaction.window_number = None,
        }
        let path = write_paginated_rollout(
            home.path(),
            "2025-01-03T13-00-01",
            uuid,
            [
                user_message("older history"),
                compacted("older usable compaction", Some(Vec::new())),
                unsupported,
            ],
        );

        assert_reverse_scan_matches_full_history(home.path(), path.as_path()).await;
    }
}

#[tokio::test]
async fn returns_scanned_full_history_at_bof_without_checkpoint() {
    let home = TempDir::new().expect("temp dir");
    let uuid = Uuid::from_u128(/*v*/ 1003);
    let path = write_paginated_rollout(
        home.path(),
        "2025-01-03T13-00-02",
        uuid,
        [
            turn_started("turn-1"),
            user_message("turn"),
            completed_user_message("turn-1", "turn"),
            turn_context(home.path(), "turn-1"),
            turn_complete("turn-1"),
        ],
    );

    assert_reverse_scan_matches_full_history(home.path(), path.as_path()).await;
}

#[tokio::test]
async fn lineage_retains_legacy_source_runtime_before_replacing_session_metadata() {
    use codex_protocol::external_executor::ModelExecutor;
    use codex_protocol::protocol::TurnModelRuntime;

    for (provider, explicit_runtime, copied_fork) in [
        (Some("source-provider"), false, false),
        (None, false, false),
        (Some("creation-provider"), true, false),
        (Some("child-provider"), false, true),
    ] {
        let home = TempDir::new().expect("temp dir");
        let root_uuid = Uuid::from_u128(2101);
        let root_id = ThreadId::from_string(&root_uuid.to_string()).unwrap();
        let expected_runtime = if explicit_runtime {
            Some(TurnModelRuntime {
                model_provider: "selected-provider".to_owned(),
                model_executor: ModelExecutor::Codex,
                model_context_window: Some(256_000),
            })
        } else if copied_fork {
            None
        } else {
            provider.map(|provider| TurnModelRuntime {
                model_provider: provider.to_owned(),
                model_executor: ModelExecutor::Antigravity,
                model_context_window: None,
            })
        };
        let RolloutItem::TurnContext(mut source_context) = turn_context(home.path(), "source")
        else {
            unreachable!();
        };
        source_context.model_runtime = explicit_runtime.then(|| expected_runtime.clone()).flatten();
        let root_path = write_ordinaled_paginated_rollout(
            home.path(),
            "2025-01-03T13-02-00",
            root_uuid,
            [
                turn_started("source"),
                user_message("source work"),
                RolloutItem::TurnContext(source_context),
                turn_complete("source"),
            ],
        );
        edit_session_meta(&root_path, |meta| {
            meta["model_provider"] = serde_json::json!(provider);
            meta["model_executor"] = serde_json::json!("antigravity");
            if copied_fork {
                meta["forked_from_id"] = serde_json::json!(ThreadId::default());
            }
        });
        let original_root = std::fs::read(&root_path).unwrap();
        let child_uuid = Uuid::from_u128(2102);
        let child_id = ThreadId::from_string(&child_uuid.to_string()).unwrap();
        let child_path =
            write_ordinaled_paginated_rollout(home.path(), "2025-01-03T13-02-01", child_uuid, []);
        set_history_base(&child_path, history_position(&root_path, root_id, 5));
        let store = LocalThreadStore::new(test_config(home.path()), None);
        let context = store
            .load_latest_model_context(LoadThreadHistoryParams {
                thread_id: child_id,
                include_archived: false,
            })
            .await
            .unwrap();

        let RolloutItem::SessionMeta(meta) = &context.items[0] else {
            panic!("canonical header");
        };
        assert_eq!(meta.meta.id, child_id);
        assert_eq!(meta.meta.model_provider.as_deref(), Some("test-provider"));
        let restored_runtime = context
            .items
            .iter()
            .find_map(|item| match item {
                RolloutItem::TurnContext(context) => Some(context.model_runtime.clone()),
                _ => None,
            })
            .unwrap();
        assert_eq!(restored_runtime, expected_runtime);
        assert_eq!(std::fs::read(root_path).unwrap(), original_root);
    }
}

#[tokio::test]
async fn lineage_recovers_owned_runtime_before_compaction_without_replaying_older_history() {
    for (unowned, previous_settings) in [(false, false), (true, false), (false, true)] {
        let home = TempDir::new().expect("temp dir");
        let source_uuid = Uuid::from_u128(2201);
        let source_id = ThreadId::from_string(&source_uuid.to_string()).unwrap();
        let settings = |owner: Option<ThreadId>, provider: &str| {
            RolloutItem::EventMsg(EventMsg::ThreadSettingsApplied(serde_json::from_value(
            serde_json::json!({
                "thread_id": owner,
                "thread_settings": {
                    "model": "test-model",
                    "model_provider_id": provider,
                    "model_executor": "antigravity",
                    "approval_policy": "on-request",
                    "approvals_reviewer": "user",
                    "permission_profile": codex_protocol::models::PermissionProfile::read_only(),
                    "cwd": home.path(),
                    "collaboration_mode": {"mode": "default", "settings": {"model": "test-model"}}
                }
            }),
        ).unwrap()))
        };
        let RolloutItem::TurnContext(mut source_context) = turn_context(home.path(), "source")
        else {
            unreachable!();
        };
        source_context.model_runtime = None;
        if previous_settings {
            source_context.model = "target-model".to_owned();
            source_context.model_runtime = Some(codex_protocol::protocol::TurnModelRuntime {
                model_provider: "target-provider".to_owned(),
                model_executor: codex_protocol::external_executor::ModelExecutor::Codex,
                model_context_window: Some(128_000),
            });
        }
        let mut checkpoint = compacted("source summary", Some(Vec::new()));
        if previous_settings && let RolloutItem::Compacted(checkpoint) = &mut checkpoint {
            checkpoint.resume_metadata = Some(
                serde_json::from_value(serde_json::json!({
                    "previous_turn_settings": {"model": "test-model"}
                }))
                .unwrap(),
            );
        }
        let source_path = write_ordinaled_paginated_rollout(
            home.path(),
            "2025-01-03T13-03-00",
            source_uuid,
            [
                settings((!unowned).then_some(source_id), "selected-provider"),
                settings(Some(ThreadId::default()), "foreign-provider"),
                turn_started("source"),
                user_message("older history"),
                checkpoint.clone(),
                RolloutItem::TurnContext(source_context.clone()),
                turn_complete("source"),
                settings(Some(source_id), "future-provider"),
            ],
        );
        let child_uuid = Uuid::from_u128(2202);
        let child_id = ThreadId::from_string(&child_uuid.to_string()).unwrap();
        let child_path =
            write_ordinaled_paginated_rollout(home.path(), "2025-01-03T13-03-01", child_uuid, []);
        set_history_base(&child_path, history_position(&source_path, source_id, 9));
        let store = LocalThreadStore::new(test_config(home.path()), None);
        let context = store
            .load_latest_model_context(LoadThreadHistoryParams {
                thread_id: child_id,
                include_archived: false,
            })
            .await
            .unwrap();
        let expected_source_runtime = Some(codex_protocol::protocol::TurnModelRuntime {
            model_provider: "selected-provider".to_owned(),
            model_executor: codex_protocol::external_executor::ModelExecutor::Antigravity,
            model_context_window: None,
        });
        if previous_settings {
            let RolloutItem::Compacted(checkpoint) = &mut checkpoint else {
                unreachable!();
            };
            checkpoint
                .resume_metadata
                .as_mut()
                .unwrap()
                .previous_turn_settings
                .as_mut()
                .unwrap()
                .model_runtime = expected_source_runtime;
        } else {
            source_context.model_runtime = expected_source_runtime;
        }
        assert_eq!(
            serde_json::to_value(&context.items[1..]).unwrap(),
            serde_json::to_value([
                checkpoint,
                RolloutItem::TurnContext(source_context),
                turn_complete("source"),
                settings(Some(source_id), "future-provider"),
            ])
            .unwrap()
        );
    }
}

#[tokio::test]
async fn checkpoint_runtime_uses_actual_turn_before_same_model_provider_change() {
    use codex_protocol::external_executor::ModelExecutor;
    use codex_protocol::protocol::TurnModelRuntime;

    for legacy_context in [false, true] {
        let home = TempDir::new().expect("temp dir");
        let source_uuid = Uuid::from_u128(2301);
        let source_id = ThreadId::from_string(&source_uuid.to_string()).unwrap();
        let settings = |provider: &str| {
            RolloutItem::EventMsg(EventMsg::ThreadSettingsApplied(
                serde_json::from_value(serde_json::json!({
                    "thread_id": source_id,
                    "thread_settings": {
                        "model": "test-model",
                        "model_provider_id": provider,
                        "model_executor": "codex",
                        "approval_policy": "on-request",
                        "approvals_reviewer": "user",
                        "permission_profile": codex_protocol::models::PermissionProfile::read_only(),
                        "cwd": home.path(),
                        "collaboration_mode": {"mode": "default", "settings": {"model": "test-model"}}
                    }
                }))
                .unwrap(),
            ))
        };
        let expected_runtime = TurnModelRuntime {
            model_provider: "source-provider".to_owned(),
            model_executor: ModelExecutor::Codex,
            model_context_window: (!legacy_context).then_some(256_000),
        };
        let RolloutItem::TurnContext(mut source_context) = turn_context(home.path(), "source")
        else {
            unreachable!();
        };
        source_context.model_runtime = (!legacy_context).then(|| expected_runtime.clone());
        let mut checkpoint = compacted("source summary", Some(Vec::new()));
        let RolloutItem::Compacted(compaction) = &mut checkpoint else {
            unreachable!();
        };
        compaction.resume_metadata = Some(
            serde_json::from_value(serde_json::json!({
                "previous_turn_settings": {"model": "test-model"}
            }))
            .unwrap(),
        );
        write_ordinaled_paginated_rollout(
            home.path(),
            "2025-01-03T13-04-00",
            source_uuid,
            [
                settings("source-provider"),
                RolloutItem::TurnContext(source_context),
                settings("target-provider"),
                checkpoint,
            ],
        );
        let store = LocalThreadStore::new(test_config(home.path()), None);
        let context = store
            .load_latest_model_context(LoadThreadHistoryParams {
                thread_id: source_id,
                include_archived: false,
            })
            .await
            .unwrap();
        assert_eq!(context.items.len(), 2);
        let RolloutItem::Compacted(checkpoint) = &context.items[1] else {
            panic!("checkpoint only, without earlier history");
        };
        assert_eq!(
            checkpoint
                .resume_metadata
                .as_ref()
                .unwrap()
                .previous_turn_settings
                .as_ref()
                .unwrap()
                .model_runtime,
            Some(expected_runtime),
        );
    }
}

#[tokio::test]
async fn replays_nested_archived_lineage_from_frozen_prefix() {
    let home = TempDir::new().expect("temp dir");
    let root_uuid = Uuid::from_u128(/*v*/ 2001);
    let root_id = ThreadId::from_string(&root_uuid.to_string()).expect("root id");
    let root_path = write_ordinaled_paginated_rollout(
        home.path(),
        "2025-01-03T13-01-00",
        root_uuid,
        [
            user_message("root before checkpoint"),
            compacted("root checkpoint", Some(Vec::new())),
            turn_started("root-excluded"),
            user_message("root after cutoff"),
        ],
    );
    let archived_root = home
        .path()
        .join("archived_sessions")
        .join(root_path.file_name().expect("root filename"));
    std::fs::create_dir_all(archived_root.parent().expect("archive parent"))
        .expect("create archive directory");
    std::fs::rename(root_path, &archived_root).expect("archive root rollout");

    let middle_uuid = Uuid::from_u128(/*v*/ 2002);
    let middle_id = ThreadId::from_string(&middle_uuid.to_string()).expect("middle id");
    let middle_path = write_ordinaled_paginated_rollout(
        home.path(),
        "2025-01-03T13-01-01",
        middle_uuid,
        [
            turn_started("middle-turn"),
            user_message("middle inherited"),
            completed_user_message("middle-turn", "middle inherited"),
            turn_context(home.path(), "middle-turn"),
            turn_complete("middle-turn"),
        ],
    );
    set_history_base(
        middle_path.as_path(),
        history_position(
            archived_root.as_path(),
            root_id,
            /*end_ordinal_exclusive*/ 3,
        ),
    );

    let child_uuid = Uuid::from_u128(/*v*/ 2003);
    let child_id = ThreadId::from_string(&child_uuid.to_string()).expect("child id");
    let child_path = write_ordinaled_paginated_rollout(
        home.path(),
        "2025-01-03T13-01-02",
        child_uuid,
        [
            turn_started("child-turn"),
            user_message("child local"),
            completed_user_message("child-turn", "child local"),
            turn_context(home.path(), "child-turn"),
            turn_complete("child-turn"),
        ],
    );
    set_history_base(
        child_path.as_path(),
        history_position(
            middle_path.as_path(),
            middle_id,
            /*end_ordinal_exclusive*/ 6,
        ),
    );
    let store = LocalThreadStore::new(test_config(home.path()), /*state_db*/ None);

    let context = store
        .load_latest_model_context(LoadThreadHistoryParams {
            thread_id: child_id,
            include_archived: false,
        })
        .await
        .expect("load lineage model context");

    assert!(matches!(
        context.items.first(),
        Some(RolloutItem::SessionMeta(meta)) if meta.meta.id == child_id
    ));
    let child_meta = codex_rollout::read_session_meta_line(child_path.as_path())
        .await
        .expect("read child metadata");
    let expected = vec![
        RolloutItem::SessionMeta(child_meta),
        compacted("root checkpoint", Some(Vec::new())),
        turn_started("middle-turn"),
        user_message("middle inherited"),
        completed_user_message("middle-turn", "middle inherited"),
        turn_context(home.path(), "middle-turn"),
        turn_complete("middle-turn"),
        turn_started("child-turn"),
        user_message("child local"),
        completed_user_message("child-turn", "child local"),
        turn_context(home.path(), "child-turn"),
        turn_complete("child-turn"),
    ];
    assert_eq!(
        serde_json::to_value(&context.items).expect("serialize context"),
        serde_json::to_value(&expected).expect("serialize expected context")
    );
    // The same frozen lineage must replay from compressed files, without materializing or
    // accidentally including the archived root's records after the inherited cutoff.
    for path in [&archived_root, &middle_path, &child_path] {
        let input = std::fs::File::open(path).expect("open rollout");
        let output = std::fs::File::create(path.with_extension("jsonl.zst"))
            .expect("create compressed rollout");
        zstd::stream::copy_encode(input, output, /*level*/ 3).expect("compress rollout");
        std::fs::remove_file(path).expect("remove plain rollout");
    }
    let compressed_context = store
        .load_latest_model_context(LoadThreadHistoryParams {
            thread_id: child_id,
            include_archived: false,
        })
        .await
        .expect("load compressed lineage model context");
    assert_eq!(
        serde_json::to_value(compressed_context.items).expect("serialize compressed context"),
        serde_json::to_value(expected).expect("serialize expected context")
    );
    assert!(
        [archived_root, middle_path, child_path]
            .iter()
            .all(|path| !path.exists())
    );
}

fn write_paginated_rollout<const N: usize>(
    home: &Path,
    timestamp: &str,
    uuid: Uuid,
    items: [RolloutItem; N],
) -> PathBuf {
    let path =
        write_session_file_with_history_mode(home, timestamp, uuid, ThreadHistoryMode::Paginated)
            .expect("write session file");
    append_items(path.as_path(), items);
    path
}

fn write_ordinaled_paginated_rollout<const N: usize>(
    home: &Path,
    timestamp: &str,
    uuid: Uuid,
    items: [RolloutItem; N],
) -> PathBuf {
    let path =
        write_session_file_with_history_mode(home, timestamp, uuid, ThreadHistoryMode::Paginated)
            .expect("write session file");
    let mut file = OpenOptions::new()
        .append(true)
        .open(path.as_path())
        .expect("open session file");
    for (index, item) in items.into_iter().enumerate() {
        let line = RolloutLine {
            timestamp: "2025-01-03T13:00:01Z".to_string(),
            ordinal: Some(u64::try_from(index).expect("fixture index fits u64") + 1),
            item,
        };
        writeln!(
            file,
            "{}",
            serde_json::to_string(&line).expect("serialize line")
        )
        .expect("append rollout line");
    }
    path
}

fn set_history_base(path: &Path, history_base: HistoryPosition) {
    edit_session_meta(path, |meta| {
        meta["history_base"] = serde_json::to_value(history_base).expect("serialize history base");
    });
}

fn edit_session_meta(path: &Path, edit: impl FnOnce(&mut serde_json::Value)) {
    let contents = std::fs::read_to_string(path).expect("read rollout");
    let mut lines = contents.lines();
    let mut head: serde_json::Value =
        serde_json::from_str(lines.next().expect("session meta line")).expect("parse head");
    edit(&mut head["payload"]);
    let mut updated = serde_json::to_string(&head).expect("serialize head");
    for line in lines {
        updated.push('\n');
        updated.push_str(line);
    }
    updated.push('\n');
    std::fs::write(path, updated).expect("write history base");
}

fn history_position(
    path: &Path,
    thread_id: ThreadId,
    end_ordinal_exclusive: u64,
) -> HistoryPosition {
    HistoryPosition {
        thread_id,
        end_ordinal_exclusive,
        end_byte_offset: rollout_end_byte_offset(path, end_ordinal_exclusive),
    }
}

fn rollout_end_byte_offset(path: &Path, end_ordinal_exclusive: u64) -> u64 {
    let contents = std::fs::read(path).expect("read rollout");
    let mut byte_offset = 0_u64;
    for line in contents.split_inclusive(|byte| *byte == b'\n') {
        let parsed = codex_rollout::parse_rollout_line_bytes(line)
            .expect("parse rollout line for byte offset");
        if parsed.ordinal == Some(end_ordinal_exclusive) {
            return byte_offset;
        }
        byte_offset += u64::try_from(line.len()).expect("line length fits u64");
    }
    byte_offset
}

async fn assert_reverse_scan_matches_full_history(home: &Path, path: &Path) {
    let session_meta = codex_rollout::read_session_meta_line(path)
        .await
        .expect("read session metadata");
    let store = LocalThreadStore::new(test_config(home), /*state_db*/ None);
    let items = store
        .load_latest_model_context(LoadThreadHistoryParams {
            thread_id: session_meta.meta.id,
            include_archived: false,
        })
        .await
        .expect("scan model context")
        .items;
    let full_items = read_thread::load_history_items(path)
        .await
        .expect("load full history");

    assert_eq!(
        serde_json::to_value(items).expect("serialize scanned items"),
        serde_json::to_value(full_items).expect("serialize full items")
    );
}

fn append_items(path: &Path, items: impl IntoIterator<Item = RolloutItem>) {
    let mut file = OpenOptions::new()
        .append(true)
        .open(path)
        .expect("open session file");
    for item in items {
        let line = RolloutLine {
            timestamp: "2025-01-03T13:00:01Z".to_string(),
            ordinal: None,
            item,
        };
        writeln!(
            file,
            "{}",
            serde_json::to_string(&line).expect("serialize line")
        )
        .expect("append rollout line");
    }
}

fn turn_started(turn_id: &str) -> RolloutItem {
    RolloutItem::EventMsg(EventMsg::TurnStarted(TurnStartedEvent {
        turn_attribution: None,
        turn_id: turn_id.to_string(),
        root_turn_id: None,
        trace_id: None,
        started_at: None,
        model_context_window: Some(128_000),
        collaboration_mode_kind: Default::default(),
    }))
}

fn turn_complete(turn_id: &str) -> RolloutItem {
    RolloutItem::EventMsg(EventMsg::TurnComplete(TurnCompleteEvent {
        root_turn_id: None,
        turn_id: turn_id.to_string(),
        last_agent_message: None,
        error: None,
        started_at: None,
        completed_at: None,
        duration_ms: None,
        time_to_first_token_ms: None,
    }))
}

fn user_message(message: &str) -> RolloutItem {
    RolloutItem::ResponseItem(
        ResponseItem::Message {
            status: None,
            encrypted_content: None,
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: message.to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        }
        .into(),
    )
}

fn contextual_user_message() -> RolloutItem {
    user_message("<environment_context>context only</environment_context>")
}

fn completed_user_message(turn_id: &str, message: &str) -> RolloutItem {
    RolloutItem::EventMsg(EventMsg::ItemCompleted(ItemCompletedEvent {
        thread_id: codex_protocol::ThreadId::from_string("00000000-0000-0000-0000-000000000000")
            .expect("fixture thread id"),
        turn_id: turn_id.to_string(),
        item: TurnItem::UserMessage(UserMessageItem {
            id: format!("user-{turn_id}"),
            client_id: None,
            content: vec![UserInput::Text {
                text: message.to_string(),
                text_elements: Vec::new(),
            }],
        }),
        started_at_ms: Some(0),
        completed_at_ms: 0,
    }))
}

fn turn_context(root: &Path, turn_id: &str) -> RolloutItem {
    RolloutItem::TurnContext(TurnContextItem {
        turn_id: Some(turn_id.to_string()),
        root_turn_id: None,
        disabled_plugin_ids: None,
        cwd: serde_json::from_value(serde_json::json!(root)).expect("absolute cwd"),
        approval_policy: AskForApproval::Never,
        approvals_reviewer: None,
        sandbox_policy: SandboxPolicy::new_read_only_policy(),
        permission_profile: None,
        active_permission_profile: None,
        file_system_sandbox_policy: None,
        model: "test-model".to_string(),
        model_runtime: Some(codex_protocol::protocol::TurnModelRuntime {
            model_provider: "test-provider".to_owned(),
            model_executor: codex_protocol::external_executor::ModelExecutor::Codex,
            model_context_window: None,
        }),
        comp_hash: None,
        collaboration_mode: None,
        multi_agent_version: None,
        realtime_active: None,
        cyber_access_program: None,
        effort: None,
        summary: Some(ReasoningSummary::Auto),
    })
}

fn compacted(message: &str, replacement_history: Option<Vec<ResponseItem>>) -> RolloutItem {
    RolloutItem::Compacted(CompactedItem {
        message: message.to_string(),
        replacement_history: replacement_history
            .map(|items| items.into_iter().map(Into::into).collect()),
        retained_context: None,
        guardian_history: None,
        mcp_resource_origins: None,
        window_number: Some(1),
        first_window_id: None,
        previous_window_id: None,
        window_id: None,
        compaction_response_id: None,
        latest_token_usage_record: None,
        resume_metadata: None,
    })
}
