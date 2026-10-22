fn main() {
    // SQLx tracks existing migration files; watch directories for newly added migrations too.
    for directory in [
        "migrations",
        "logs_migrations",
        "goals_migrations",
        "memory_migrations",
        "queue_migrations",
        "thread_history_migrations",
    ] {
        println!("cargo:rerun-if-changed={directory}");
    }
}
