-- Discodex projection semantics use this namespaced version so future upstream
-- sequential migrations remain independent. Preserve all existing rows and offsets;
-- the native projector replays a version-zero view before advancing its checkpoint.
ALTER TABLE thread_history_projection_state
    ADD COLUMN projection_version INTEGER NOT NULL DEFAULT 0;
