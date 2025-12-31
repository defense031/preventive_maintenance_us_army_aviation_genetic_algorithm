"""SQL database integration for detailed daily data collection.

Provides SQLite database operations for storing and retrieving simulation data.
Schema matches R version's detailed_daily_operations.csv structure.
"""

import sqlite3
import os
from typing import List, Dict, Optional
from pathlib import Path


def initialize_database(db_path: str) -> sqlite3.Connection:
    """Initialize SQLite database for simulation data collection.

    Args:
        db_path: Path to database file

    Returns:
        Database connection object
    """
    # Create directory if needed
    db_dir = os.path.dirname(db_path)
    if db_dir:
        Path(db_dir).mkdir(parents=True, exist_ok=True)

    # Connect to database (creates if doesn't exist)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")  # Enable Write-Ahead Logging for performance

    # Create tables
    create_tables(conn)

    return conn


def create_tables(conn: sqlite3.Connection) -> None:
    """Create episodes and daily_data tables if they don't exist.

    Args:
        conn: Database connection
    """
    cursor = conn.cursor()

    # Episodes table - stores episode-level summary metrics
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS episodes (
            episode_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            episode_number INTEGER NOT NULL,
            sim_days INTEGER,
            mean_or REAL,
            final_or REAL,
            mission_success_rate REAL,
            total_flight_hours REAL,
            total_inflight_failures INTEGER,
            total_nmc_days INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Daily data table - stores aircraft-level daily data (matches R version exactly)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_data (
            daily_id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id INTEGER NOT NULL,
            day INTEGER NOT NULL,
            aircraft_id INTEGER NOT NULL,
            decision TEXT,
            true_rul REAL,
            observed_rul REAL,
            total_flight_hours REAL,
            todays_flight_hours REAL,
            hours_since_minor_phase REAL,
            hours_since_major_phase REAL,
            status TEXT,
            flew_today INTEGER,
            flight_failure_today INTEGER,
            maintenance_started TEXT,
            maintenance_completed TEXT,
            days_in_maintenance INTEGER,
            queue_position TEXT,
            tokens_remaining_preventive INTEGER,
            tokens_remaining_minor_phase INTEGER,
            tokens_remaining_major_phase INTEGER,
            episode INTEGER,
            session_id INTEGER,
            FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
        )
    """)

    # Create indices for faster queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_daily_episode
        ON daily_data(episode_id, day, aircraft_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_episode_session
        ON episodes(session_id, episode_number)
    """)

    conn.commit()


def create_episode_record(
    conn: sqlite3.Connection,
    session_id: int,
    episode_number: int,
    sim_days: int
) -> int:
    """Create a new episode record and return its ID.

    Args:
        conn: Database connection
        session_id: Session identifier
        episode_number: Episode number within session
        sim_days: Number of simulation days

    Returns:
        episode_id (auto-generated)
    """
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO episodes (session_id, episode_number, sim_days)
        VALUES (?, ?, ?)
    """, (session_id, episode_number, sim_days))
    conn.commit()

    return cursor.lastrowid


def update_episode_metrics(
    conn: sqlite3.Connection,
    episode_id: int,
    metrics: Dict
) -> None:
    """Update episode record with final metrics.

    Args:
        conn: Database connection
        episode_id: Episode ID to update
        metrics: Dict with final episode metrics
    """
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE episodes
        SET mean_or = ?,
            final_or = ?,
            mission_success_rate = ?,
            total_flight_hours = ?,
            total_inflight_failures = ?,
            total_nmc_days = ?
        WHERE episode_id = ?
    """, (
        metrics.get("mean_or"),
        metrics.get("final_or"),
        metrics.get("total_mission_success_rate"),
        metrics.get("total_flight_hours"),
        metrics.get("total_inflight_failures"),
        metrics.get("total_nmc_days"),
        episode_id
    ))
    conn.commit()


def write_daily_data(conn: sqlite3.Connection, records: List[Dict]) -> None:
    """Write daily aircraft records to database.

    Args:
        conn: Database connection
        records: List of daily aircraft state dicts (one per aircraft)
    """
    if not records:
        return

    cursor = conn.cursor()

    # Batch insert all records for this day
    cursor.executemany("""
        INSERT INTO daily_data (
            episode_id, day, aircraft_id, decision,
            true_rul, observed_rul, total_flight_hours, todays_flight_hours,
            hours_since_minor_phase, hours_since_major_phase, status,
            flew_today, flight_failure_today, maintenance_started, maintenance_completed,
            days_in_maintenance, queue_position,
            tokens_remaining_preventive, tokens_remaining_minor_phase, tokens_remaining_major_phase,
            episode, session_id
        ) VALUES (
            :episode_id, :day, :aircraft_id, :decision,
            :true_rul, :observed_rul, :total_flight_hours, :todays_flight_hours,
            :hours_since_minor_phase, :hours_since_major_phase, :status,
            :flew_today, :flight_failure_today, :maintenance_started, :maintenance_completed,
            :days_in_maintenance, :queue_position,
            :tokens_remaining_preventive, :tokens_remaining_minor_phase, :tokens_remaining_major_phase,
            :episode, :session_id
        )
    """, records)

    conn.commit()


def close_database(conn: Optional[sqlite3.Connection]) -> None:
    """Close database connection safely.

    Args:
        conn: Database connection (None-safe)
    """
    if conn:
        conn.close()


def get_database_stats(db_path: str) -> Dict:
    """Get database statistics for verification.

    Args:
        db_path: Path to database file

    Returns:
        Dict with episode count, row count, etc.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM episodes")
    episode_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM daily_data")
    row_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT session_id) FROM episodes")
    session_count = cursor.fetchone()[0]

    conn.close()

    return {
        "episodes": episode_count,
        "daily_data_rows": row_count,
        "sessions": session_count,
    }
