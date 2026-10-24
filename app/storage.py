
"""SQLite persistence, owner isolation, atomic state transitions, and decision cache."""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = os.getenv(
    "DATABASE_PATH",
    os.getenv("INVOICE_DB_PATH", "./data/invoice_a2a.db"),
)


def _connect() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=10, isolation_level=None)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA busy_timeout=10000")
    db.execute("PRAGMA foreign_keys=ON")
    return db


async def init_db() -> None:
    with _connect() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            principal TEXT NOT NULL,
            context_id TEXT NOT NULL,
            state TEXT NOT NULL,
            batch_id TEXT,
            history_json TEXT NOT NULL,
            artifacts_json TEXT NOT NULL,
            proposals_json TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_tasks_principal
            ON tasks(principal);

        CREATE TABLE IF NOT EXISTS idempotency (
            principal TEXT NOT NULL,
            message_id TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            task_id TEXT NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY (principal, message_id)
        );

        CREATE TABLE IF NOT EXISTS decision_cache (
            content_hash TEXT PRIMARY KEY,
            decision_json TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        """)


def _row_to_task(row: sqlite3.Row) -> Dict[str, Any]:
    task = {
        "id": row["task_id"],
        "contextId": row["context_id"],
        "status": {"state": row["state"]},
        "history": json.loads(row["history_json"] or "[]"),
        "artifacts": json.loads(row["artifacts_json"] or "[]"),
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
    }
    if row["batch_id"]:
        task["batchId"] = row["batch_id"]
    if row["proposals_json"]:
        task["_proposals"] = json.loads(row["proposals_json"])
    return task


async def get_task(
    principal: str, task_id: str
) -> Optional[Dict[str, Any]]:
    with _connect() as db:
        row = db.execute(
            "SELECT * FROM tasks WHERE task_id=? AND principal=?",
            (task_id, principal),
        ).fetchone()
        return _row_to_task(row) if row else None


async def list_tasks(principal: str) -> List[Dict[str, Any]]:
    with _connect() as db:
        rows = db.execute(
            """
            SELECT * FROM tasks
            WHERE principal=?
            ORDER BY created_at DESC LIMIT 100
            """,
            (principal,),
        ).fetchall()
        return [_row_to_task(row) for row in rows]


async def save_task(
    task: Dict[str, Any], principal: str
) -> None:
    now = time.time()
    with _connect() as db:
        db.execute(
            """
            INSERT INTO tasks (
                task_id, principal, context_id, state, batch_id,
                history_json, artifacts_json, proposals_json,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                state=excluded.state,
                history_json=excluded.history_json,
                artifacts_json=excluded.artifacts_json,
                proposals_json=excluded.proposals_json,
                updated_at=excluded.updated_at
            WHERE tasks.principal=excluded.principal
            """,
            (
                task["id"],
                principal,
                task["contextId"],
                task["status"]["state"],
                task.get("batchId"),
                json.dumps(task.get("history", []), ensure_ascii=False),
                json.dumps(task.get("artifacts", []), ensure_ascii=False),
                json.dumps(task.get("_proposals"), ensure_ascii=False)
                if task.get("_proposals") else None,
                task.get("createdAt", now),
                now,
            ),
        )


async def reserve_initial(
    task: Dict[str, Any],
    principal: str,
    message_id: str,
    content_hash: str,
):
    """Atomically reserve an initial message and task."""
    with _connect() as db:
        db.execute("BEGIN IMMEDIATE")
        try:
            row = db.execute(
                """
                SELECT task_id, content_hash FROM idempotency
                WHERE principal=? AND message_id=?
                """,
                (principal, message_id),
            ).fetchone()

            if row:
                db.execute("COMMIT")
                return False, row["content_hash"], row["task_id"]

            now = time.time()
            db.execute(
                """
                INSERT INTO tasks (
                    task_id, principal, context_id, state, batch_id,
                    history_json, artifacts_json, proposals_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task["id"], principal, task["contextId"],
                    task["status"]["state"], task.get("batchId"),
                    json.dumps(task.get("history", []), ensure_ascii=False),
                    json.dumps(task.get("artifacts", []), ensure_ascii=False),
                    None, now, now,
                ),
            )
            db.execute(
                """
                INSERT INTO idempotency (
                    principal, message_id, content_hash, task_id, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (principal, message_id, content_hash, task["id"], now),
            )
            db.execute("COMMIT")
            return True, content_hash, task["id"]
        except Exception:
            if db.in_transaction:
                db.execute("ROLLBACK")
            raise


async def transition_task(
    task: Dict[str, Any],
    principal: str,
    expected_state: str,
) -> bool:
    """Change task state only when its current state matches."""
    with _connect() as db:
        cur = db.execute(
            """
            UPDATE tasks
            SET state=?, history_json=?, artifacts_json=?,
                proposals_json=?, updated_at=?
            WHERE task_id=? AND principal=? AND state=?
            """,
            (
                task["status"]["state"],
                json.dumps(task.get("history", []), ensure_ascii=False),
                json.dumps(task.get("artifacts", []), ensure_ascii=False),
                json.dumps(task.get("_proposals"), ensure_ascii=False)
                if task.get("_proposals") else None,
                time.time(), task["id"], principal, expected_state,
            ),
        )
        return cur.rowcount == 1


async def check_idempotency(
    principal: str, message_id: str, content_hash: str
):
    with _connect() as db:
        row = db.execute(
            """
            SELECT task_id, content_hash FROM idempotency
            WHERE principal=? AND message_id=?
            """,
            (principal, message_id),
        ).fetchone()
        return (row["task_id"], row["content_hash"]) if row else (None, None)


async def record_idempotency(
    principal: str,
    message_id: str,
    content_hash: str,
    task_id: str,
) -> None:
    with _connect() as db:
        db.execute(
            """
            INSERT OR IGNORE INTO idempotency (
                principal, message_id, content_hash, task_id, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (principal, message_id, content_hash, task_id, time.time()),
        )


async def commit_continuation(
    task: Dict[str, Any],
    principal: str,
    message_id: str,
    content_hash: str,
    expected_state: str,
) -> bool:
    """Atomically commit a task transition and its idempotency record."""
    with _connect() as db:
        db.execute("BEGIN IMMEDIATE")
        try:
            existing = db.execute(
                """
                SELECT content_hash, task_id FROM idempotency
                WHERE principal=? AND message_id=?
                """,
                (principal, message_id),
            ).fetchone()

            if existing:
                db.execute("ROLLBACK")
                return False

            cur = db.execute(
                """
                UPDATE tasks
                SET state=?, history_json=?, artifacts_json=?,
                    proposals_json=?, updated_at=?
                WHERE task_id=? AND principal=? AND state=?
                """,
                (
                    task["status"]["state"],
                    json.dumps(task.get("history", []), ensure_ascii=False),
                    json.dumps(task.get("artifacts", []), ensure_ascii=False),
                    json.dumps(task.get("_proposals"), ensure_ascii=False)
                    if task.get("_proposals") else None,
                    time.time(), task["id"], principal, expected_state,
                ),
            )

            if cur.rowcount != 1:
                db.execute("ROLLBACK")
                return False

            db.execute(
                """
                INSERT INTO idempotency (
                    principal, message_id, content_hash, task_id, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (principal, message_id, content_hash, task["id"], time.time()),
            )
            db.execute("COMMIT")
            return True
        except Exception:
            if db.in_transaction:
                db.execute("ROLLBACK")
            raise


async def get_cached_decision(content_hash: str):
    with _connect() as db:
        row = db.execute(
            "SELECT decision_json FROM decision_cache WHERE content_hash=?",
            (content_hash,),
        ).fetchone()
        return json.loads(row["decision_json"]) if row else None


async def put_cached_decision(
    content_hash: str, decision: Dict[str, Any]
) -> None:
    with _connect() as db:
        db.execute(
            """
            INSERT OR REPLACE INTO decision_cache
                (content_hash, decision_json, created_at)
            VALUES (?, ?, ?)
            """,
            (
                content_hash,
                json.dumps(decision, ensure_ascii=False),
                time.time(),
            ),
        )


def new_task_id() -> str:
    return str(uuid.uuid4())


def new_context_id() -> str:
    return str(uuid.uuid4())


def new_action_id() -> str:
    return uuid.uuid4().hex
