"""
database.py — Persistent Storage Layer for the Cognitive Immune System

Mathematical Foundation:
    The CC-DAG (Contamination Causal Directed Acyclic Graph) requires persistent
    storage to maintain immunological memory across sessions. This module implements
    the relational schema that backs the graph structure G = (V, E) where:
        V = {v_i | v_i is a contamination event with attributes (text, cause, score, τ)}
        E = {(v_i, v_j) | v_i causally precedes v_j in the contamination chain}

    The safe_registry table implements the tolerance set T:
        T = {c | Wikipedia_confidence(c) > 0.85}
    Claims in T are never quarantined, preventing autoimmune false positives.

Author: Muhammad Saad, Independent Researcher, Pakistan
"""

import sqlite3
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("cis.database")

DATABASE_URL: str = os.getenv("DATABASE_URL", "./cis.db")


def _get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Establish a SQLite connection with WAL mode for concurrent read performance."""
    path = db_path or DATABASE_URL
    try:
        conn = sqlite3.connect(path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logger.error("Failed to connect to database at %s: %s", path, e)
        raise


def init_db(db_path: Optional[str] = None) -> None:
    """Initialize database schema — idempotent, safe to call on every startup."""
    conn = _get_connection(db_path)
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS contamination_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_text  TEXT NOT NULL,
                score       REAL NOT NULL,
                cause       TEXT DEFAULT '',
                session_id  TEXT DEFAULT '',
                timestamp   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS causal_edges (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id   INTEGER NOT NULL,
                target_id   INTEGER NOT NULL,
                relation    TEXT NOT NULL DEFAULT 'caused_contamination',
                timestamp   TEXT NOT NULL,
                FOREIGN KEY (source_id) REFERENCES contamination_events(id),
                FOREIGN KEY (target_id) REFERENCES contamination_events(id)
            );

            CREATE TABLE IF NOT EXISTS safe_registry (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_text  TEXT NOT NULL UNIQUE,
                confidence  REAL NOT NULL,
                added_at    TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS experiment_results (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id         INTEGER NOT NULL,
                question_group      TEXT NOT NULL,
                system_type         TEXT NOT NULL,
                question_text       TEXT DEFAULT '',
                answer              TEXT DEFAULT '',
                gold_answer         TEXT DEFAULT '',
                exact_match         INTEGER NOT NULL DEFAULT 0,
                contamination_rate  REAL DEFAULT 0.0,
                claims_total        INTEGER DEFAULT 0,
                claims_quarantined  INTEGER DEFAULT 0,
                latency_ms          INTEGER DEFAULT 0,
                timestamp           TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_events_session
                ON contamination_events(session_id);
            CREATE INDEX IF NOT EXISTS idx_events_score
                ON contamination_events(score);
            CREATE INDEX IF NOT EXISTS idx_edges_source
                ON causal_edges(source_id);
            CREATE INDEX IF NOT EXISTS idx_edges_target
                ON causal_edges(target_id);
            CREATE INDEX IF NOT EXISTS idx_safe_claim
                ON safe_registry(claim_text);
            CREATE INDEX IF NOT EXISTS idx_experiment_qid
                ON experiment_results(question_id, system_type);
        """)
        conn.commit()
        logger.info("Database initialized successfully at %s", db_path or DATABASE_URL)
    except sqlite3.Error as e:
        logger.error("Failed to initialize database: %s", e)
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Contamination Events
# ---------------------------------------------------------------------------

def insert_event(
    claim_text: str,
    score: float,
    cause: str = "",
    session_id: str = "",
    db_path: Optional[str] = None,
) -> int:
    """Insert a contamination event and return its row ID."""
    conn = _get_connection(db_path)
    try:
        cursor = conn.execute(
            """INSERT INTO contamination_events (claim_text, score, cause, session_id, timestamp)
               VALUES (?, ?, ?, ?, ?)""",
            (claim_text, score, cause, session_id, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        row_id: int = cursor.lastrowid  # type: ignore[assignment]
        return row_id
    except sqlite3.Error as e:
        logger.error("Failed to insert contamination event: %s", e)
        raise
    finally:
        conn.close()


def get_events(
    session_id: Optional[str] = None, db_path: Optional[str] = None
) -> list[dict]:
    """Retrieve contamination events, optionally filtered by session."""
    conn = _get_connection(db_path)
    try:
        if session_id:
            rows = conn.execute(
                "SELECT * FROM contamination_events WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM contamination_events ORDER BY id"
            ).fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        logger.error("Failed to retrieve contamination events: %s", e)
        return []
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Causal Edges
# ---------------------------------------------------------------------------

def insert_edge(
    source_id: int,
    target_id: int,
    relation: str = "caused_contamination",
    db_path: Optional[str] = None,
) -> int:
    """Insert a causal edge between two contamination events."""
    conn = _get_connection(db_path)
    try:
        cursor = conn.execute(
            """INSERT INTO causal_edges (source_id, target_id, relation, timestamp)
               VALUES (?, ?, ?, ?)""",
            (source_id, target_id, relation, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]
    except sqlite3.Error as e:
        logger.error("Failed to insert causal edge: %s", e)
        raise
    finally:
        conn.close()


def get_edges(db_path: Optional[str] = None) -> list[dict]:
    """Retrieve all causal edges from the database."""
    conn = _get_connection(db_path)
    try:
        rows = conn.execute("SELECT * FROM causal_edges ORDER BY id").fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        logger.error("Failed to retrieve causal edges: %s", e)
        return []
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Safe Registry (Tolerance Set T)
# ---------------------------------------------------------------------------

def is_safe_registered(claim_text: str, db_path: Optional[str] = None) -> bool:
    """Check if a claim exists in the tolerance set T."""
    conn = _get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT 1 FROM safe_registry WHERE claim_text = ?", (claim_text,)
        ).fetchone()
        return row is not None
    except sqlite3.Error as e:
        logger.error("Failed to check safe registry: %s", e)
        return False
    finally:
        conn.close()


def register_safe(
    claim_text: str, confidence: float, db_path: Optional[str] = None
) -> None:
    """Add a claim to the tolerance set T (safe registry)."""
    conn = _get_connection(db_path)
    try:
        conn.execute(
            """INSERT OR IGNORE INTO safe_registry (claim_text, confidence, added_at)
               VALUES (?, ?, ?)""",
            (claim_text, confidence, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        logger.debug("Registered safe claim: %.60s...", claim_text)
    except sqlite3.Error as e:
        logger.error("Failed to register safe claim: %s", e)
    finally:
        conn.close()


def get_safe_registry(db_path: Optional[str] = None) -> list[dict]:
    """Retrieve the full tolerance set T."""
    conn = _get_connection(db_path)
    try:
        rows = conn.execute("SELECT * FROM safe_registry ORDER BY id").fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        logger.error("Failed to retrieve safe registry: %s", e)
        return []
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Experiment Results
# ---------------------------------------------------------------------------

def insert_experiment_result(
    question_id: int,
    question_group: str,
    system_type: str,
    question_text: str,
    answer: str,
    gold_answer: str,
    exact_match: bool,
    contamination_rate: float,
    claims_total: int,
    claims_quarantined: int,
    latency_ms: int,
    db_path: Optional[str] = None,
) -> int:
    """Insert a single experiment result row."""
    conn = _get_connection(db_path)
    try:
        cursor = conn.execute(
            """INSERT INTO experiment_results
               (question_id, question_group, system_type, question_text, answer,
                gold_answer, exact_match, contamination_rate, claims_total,
                claims_quarantined, latency_ms, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                question_id, question_group, system_type, question_text, answer,
                gold_answer, int(exact_match), contamination_rate, claims_total,
                claims_quarantined, latency_ms, datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]
    except sqlite3.Error as e:
        logger.error("Failed to insert experiment result: %s", e)
        raise
    finally:
        conn.close()


def get_experiment_results(
    system_type: Optional[str] = None, db_path: Optional[str] = None
) -> list[dict]:
    """Retrieve experiment results, optionally filtered by system type."""
    conn = _get_connection(db_path)
    try:
        if system_type:
            rows = conn.execute(
                "SELECT * FROM experiment_results WHERE system_type = ? ORDER BY question_id",
                (system_type,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM experiment_results ORDER BY question_id"
            ).fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        logger.error("Failed to retrieve experiment results: %s", e)
        return []
    finally:
        conn.close()


def get_completed_question_ids(db_path: Optional[str] = None) -> set[tuple[int, str]]:
    """Return set of (question_id, system_type) pairs already completed — for checkpointing."""
    conn = _get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT DISTINCT question_id, system_type FROM experiment_results"
        ).fetchall()
        return {(row["question_id"], row["system_type"]) for row in rows}
    except sqlite3.Error as e:
        logger.error("Failed to retrieve completed question IDs: %s", e)
        return set()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Aggregate Statistics
# ---------------------------------------------------------------------------

def get_stats(db_path: Optional[str] = None) -> dict:
    """Compute aggregate statistics across all contamination events."""
    conn = _get_connection(db_path)
    try:
        total = conn.execute("SELECT COUNT(*) as c FROM contamination_events").fetchone()
        avg_score = conn.execute("SELECT AVG(score) as a FROM contamination_events").fetchone()
        quarantined = conn.execute(
            "SELECT COUNT(*) as c FROM contamination_events WHERE score >= ?",
            (float(os.getenv("CONTAMINATION_THRESHOLD", "0.7")),),
        ).fetchone()
        return {
            "total_analyzed": total["c"] if total else 0,
            "avg_contamination_rate": round(avg_score["a"], 4) if avg_score and avg_score["a"] else 0.0,
            "total_quarantined": quarantined["c"] if quarantined else 0,
        }
    except sqlite3.Error as e:
        logger.error("Failed to compute stats: %s", e)
        return {"total_analyzed": 0, "avg_contamination_rate": 0.0, "total_quarantined": 0}
    finally:
        conn.close()
