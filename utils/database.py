"""Database utilities for simulation data logging.

Ported from: aviation_hierarchical_sim_v2/convergence_v2/sql_integration_functions.R

Simplified schema for GA/baseline simulation (no RL agent tracking).

Tables:
- sessions: Simulation session metadata
- episodes: Episode-level summary metrics
- daily_data: Per-aircraft, per-day detailed state (optional for performance)
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import json


class SimulationDatabase:
    """SQLite database for simulation data logging.

    Provides simple interface for recording session, episode, and daily data.
    """

    def __init__(self, db_path: str, verbose: bool = False):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file
            verbose: Enable detailed logging
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        # Create connection and enable WAL mode (prevents locking)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")

        # Create schemas
        self._create_schema()

        if self.verbose:
            print(f"📊 Database initialized: {self.db_path}")

    def _create_schema(self) -> None:
        """Create database schema."""
        cursor = self.conn.cursor()

        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_name TEXT NOT NULL,
                start_timestamp DATETIME NOT NULL,
                end_timestamp DATETIME,
                total_episodes INTEGER DEFAULT 0,
                num_aircraft INTEGER,
                sim_days INTEGER,
                policy_type TEXT,
                policy_genome TEXT,
                config_yaml TEXT,
                notes TEXT
            )
        """)

        # Episodes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER REFERENCES sessions(session_id),
                session_episode INTEGER,
                episode_start DATETIME NOT NULL,
                episode_end DATETIME,
                sim_days INTEGER,
                seed INTEGER,
                final_or REAL,
                mean_or REAL,
                total_mission_success_rate REAL,
                total_flight_hours REAL,
                total_inflight_failures INTEGER,
                total_nmc_days INTEGER,
                reward REAL,
                fitness REAL
            )
        """)

        # Daily data table (optional - only if collect_detailed_data=True)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_data (
                daily_id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id INTEGER REFERENCES episodes(episode_id),
                day INTEGER,
                aircraft_id INTEGER,
                decision TEXT,
                true_rul REAL,
                observed_rul REAL,
                total_flight_hours REAL,
                todays_flight_hours REAL,
                hours_since_minor_phase REAL,
                hours_since_major_phase REAL,
                status TEXT,
                flew_today BOOLEAN,
                flight_failure_today BOOLEAN,
                maintenance_started TEXT,
                maintenance_completed TEXT,
                days_in_maintenance INTEGER,
                queue_position TEXT,
                tokens_remaining_preventive INTEGER,
                tokens_remaining_minor_phase INTEGER,
                tokens_remaining_major_phase INTEGER,
                episode INTEGER,
                session_id INTEGER
            )
        """)

        # Create indices for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_episode ON daily_data(episode_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_day ON daily_data(day)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id)")

        self.conn.commit()

    def create_session(
        self,
        session_name: str,
        num_aircraft: int,
        sim_days: int,
        policy_type: str,
        policy_genome: Optional[str] = None,
        config_dict: Optional[Dict] = None,
        notes: Optional[str] = None,
    ) -> int:
        """Create new simulation session.

        Args:
            session_name: Name of the session
            num_aircraft: Number of aircraft in fleet
            sim_days: Simulation duration in days
            policy_type: Type of policy (e.g., 'baseline', 'decision_tree', 'random')
            policy_genome: JSON string of policy genome (for GA policies)
            config_dict: Configuration dictionary (will be serialized to JSON)
            notes: Optional notes

        Returns:
            session_id
        """
        cursor = self.conn.cursor()

        config_yaml = json.dumps(config_dict) if config_dict else None

        cursor.execute(
            """
            INSERT INTO sessions (
                session_name, start_timestamp, num_aircraft, sim_days,
                policy_type, policy_genome, config_yaml, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                session_name,
                datetime.now(),
                num_aircraft,
                sim_days,
                policy_type,
                policy_genome,
                config_yaml,
                notes,
            ),
        )

        self.conn.commit()
        session_id = cursor.lastrowid

        if self.verbose:
            print(f"📝 Created session {session_id}: {session_name}")

        return session_id

    def close_session(self, session_id: int, total_episodes: int) -> None:
        """Close session and record end timestamp.

        Args:
            session_id: Session to close
            total_episodes: Total number of episodes completed
        """
        cursor = self.conn.cursor()

        cursor.execute(
            """
            UPDATE sessions
            SET end_timestamp = ?, total_episodes = ?
            WHERE session_id = ?
        """,
            (datetime.now(), total_episodes, session_id),
        )

        self.conn.commit()

        if self.verbose:
            print(f"✅ Closed session {session_id} with {total_episodes} episodes")

    def create_episode(self, session_id: int, session_episode: int, sim_days: int, seed: Optional[int] = None) -> int:
        """Create new episode entry.

        Args:
            session_id: Parent session ID
            session_episode: Episode number within session
            sim_days: Simulation duration
            seed: Random seed (if any)

        Returns:
            episode_id
        """
        cursor = self.conn.cursor()

        cursor.execute(
            """
            INSERT INTO episodes (
                session_id, session_episode, episode_start, sim_days, seed
            ) VALUES (?, ?, ?, ?, ?)
        """,
            (session_id, session_episode, datetime.now(), sim_days, seed),
        )

        self.conn.commit()
        return cursor.lastrowid

    def close_episode(self, episode_id: int, metrics: Dict) -> None:
        """Close episode and record final metrics.

        Args:
            episode_id: Episode to close
            metrics: Dict with keys: final_or, mean_or, total_mission_success_rate,
                     total_flight_hours, total_inflight_failures, total_nmc_days,
                     reward, fitness
        """
        cursor = self.conn.cursor()

        cursor.execute(
            """
            UPDATE episodes
            SET episode_end = ?,
                final_or = ?,
                mean_or = ?,
                total_mission_success_rate = ?,
                total_flight_hours = ?,
                total_inflight_failures = ?,
                total_nmc_days = ?,
                reward = ?,
                fitness = ?
            WHERE episode_id = ?
        """,
            (
                datetime.now(),
                metrics.get("final_or"),
                metrics.get("mean_or"),
                metrics.get("total_mission_success_rate"),
                metrics.get("total_flight_hours"),
                metrics.get("total_inflight_failures"),
                metrics.get("total_nmc_days"),
                metrics.get("reward"),
                metrics.get("fitness"),
                episode_id,
            ),
        )

        self.conn.commit()

    def log_daily_data(self, episode_id: int, day: int, aircraft_states: List[Dict], session_id: int, session_episode: int) -> None:
        """Log daily aircraft states.

        Args:
            episode_id: Episode ID
            day: Simulation day
            aircraft_states: List of dicts with aircraft state info
            session_id: Session ID
            session_episode: Episode number within session
        """
        cursor = self.conn.cursor()

        rows = []
        for aircraft in aircraft_states:
            rows.append(
                (
                    episode_id,
                    day,
                    aircraft["aircraft_id"],
                    aircraft.get("decision"),
                    aircraft.get("true_rul"),
                    aircraft.get("observed_rul"),
                    aircraft.get("total_flight_hours"),
                    aircraft.get("todays_flight_hours"),
                    aircraft.get("hours_since_minor_phase"),
                    aircraft.get("hours_since_major_phase"),
                    aircraft.get("status"),
                    aircraft.get("flew_today"),
                    aircraft.get("flight_failure_today"),
                    aircraft.get("maintenance_started"),
                    aircraft.get("maintenance_completed"),
                    aircraft.get("days_in_maintenance"),
                    aircraft.get("queue_position"),
                    aircraft.get("tokens_remaining_preventive"),
                    aircraft.get("tokens_remaining_minor_phase"),
                    aircraft.get("tokens_remaining_major_phase"),
                    session_episode,
                    session_id,
                )
            )

        cursor.executemany(
            """
            INSERT INTO daily_data (
                episode_id, day, aircraft_id, decision,
                true_rul, observed_rul, total_flight_hours, todays_flight_hours,
                hours_since_minor_phase, hours_since_major_phase,
                status, flew_today, flight_failure_today,
                maintenance_started, maintenance_completed, days_in_maintenance,
                queue_position, tokens_remaining_preventive,
                tokens_remaining_minor_phase, tokens_remaining_major_phase,
                episode, session_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            rows,
        )

        self.conn.commit()

    def get_episode_metrics(self, episode_id: int) -> Optional[Dict]:
        """Retrieve episode metrics.

        Args:
            episode_id: Episode ID

        Returns:
            Dict with episode metrics or None if not found
        """
        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT final_or, mean_or, total_mission_success_rate,
                   total_flight_hours, total_inflight_failures, total_nmc_days,
                   reward, fitness
            FROM episodes
            WHERE episode_id = ?
        """,
            (episode_id,),
        )

        row = cursor.fetchone()
        if row is None:
            return None

        return {
            "final_or": row[0],
            "mean_or": row[1],
            "total_mission_success_rate": row[2],
            "total_flight_hours": row[3],
            "total_inflight_failures": row[4],
            "total_nmc_days": row[5],
            "reward": row[6],
            "fitness": row[7],
        }

    def get_session_summary(self, session_id: int) -> Optional[Dict]:
        """Get session summary statistics.

        Args:
            session_id: Session ID

        Returns:
            Dict with session summary
        """
        cursor = self.conn.cursor()

        # Get session metadata
        cursor.execute(
            """
            SELECT session_name, start_timestamp, end_timestamp, total_episodes,
                   policy_type, num_aircraft, sim_days
            FROM sessions
            WHERE session_id = ?
        """,
            (session_id,),
        )

        row = cursor.fetchone()
        if row is None:
            return None

        # Get episode statistics
        cursor.execute(
            """
            SELECT AVG(mean_or), AVG(total_mission_success_rate),
                   AVG(total_flight_hours), SUM(total_inflight_failures)
            FROM episodes
            WHERE session_id = ?
        """,
            (session_id,),
        )

        stats = cursor.fetchone()

        return {
            "session_name": row[0],
            "start_timestamp": row[1],
            "end_timestamp": row[2],
            "total_episodes": row[3],
            "policy_type": row[4],
            "num_aircraft": row[5],
            "sim_days": row[6],
            "avg_mean_or": stats[0],
            "avg_mission_success_rate": stats[1],
            "avg_flight_hours": stats[2],
            "total_inflight_failures": stats[3],
        }

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()

        if self.verbose:
            print(f"📊 Database closed: {self.db_path}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
