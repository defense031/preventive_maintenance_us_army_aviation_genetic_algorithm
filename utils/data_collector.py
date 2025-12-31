"""Data collection and persistence for simulation runs.

Handles all database operations separate from simulation logic.
Provides callback-based interface for collecting data during simulation.
"""

from typing import Optional, Dict, List
from pathlib import Path
import sqlite3

from utils.sql_integration import (
    initialize_database,
    create_episode_record,
    update_episode_metrics,
    write_daily_data,
    close_database,
)
from utils.daily_data_converter import convert_state_to_daily_records


class DataCollector:
    """Collects and persists simulation data to SQLite database.

    Separates data persistence logic from simulation logic.
    Provides clean interface for runner scripts to collect data.
    """

    def __init__(
        self,
        db_path: str,
        session_id: int = 1,
        enabled: bool = True,
        verbose: bool = False
    ):
        """Initialize data collector.

        Args:
            db_path: Path to SQLite database file
            session_id: Session identifier for grouping episodes
            enabled: If False, collector is a no-op (for performance)
            verbose: Enable detailed logging
        """
        self.db_path = Path(db_path)
        self.session_id = session_id
        self.enabled = enabled
        self.verbose = verbose

        self.db_conn: Optional[sqlite3.Connection] = None
        self.current_episode_id: Optional[int] = None
        self.current_episode_number: int = 0

        if self.enabled:
            self._initialize()

    def _initialize(self) -> None:
        """Initialize database and create schema."""
        # Create parent directory if needed
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database connection
        self.db_conn = initialize_database(str(self.db_path))

        if self.verbose:
            print(f"📊 DataCollector initialized: {self.db_path}")

    def start_episode(self, episode_number: int, sim_days: int) -> int:
        """Start collecting data for a new episode.

        Args:
            episode_number: Episode number within session
            sim_days: Simulation duration in days

        Returns:
            episode_id from database
        """
        if not self.enabled or self.db_conn is None:
            return -1

        self.current_episode_number = episode_number

        # Create episode record
        self.current_episode_id = create_episode_record(
            conn=self.db_conn,
            session_id=self.session_id,
            episode_number=episode_number,
            sim_days=sim_days
        )

        if self.verbose:
            print(f"📝 Started episode {episode_number} (episode_id={self.current_episode_id})")

        return self.current_episode_id

    def collect_daily_data(
        self,
        day: int,
        state: Dict,
        actions: Dict[int, str],
        tokens_available: Dict[str, int]
    ) -> None:
        """Collect daily aircraft state data.

        Args:
            day: Current simulation day
            state: Environment state dict with aircraft list
            actions: Dict mapping aircraft_id to action string
            tokens_available: Dict with token counts by type
        """
        if not self.enabled or self.db_conn is None or self.current_episode_id is None:
            return

        # Convert state to daily records format
        # Note: tokens_available is a dict, but we pass a single value (reactive tokens)
        tokens_count = tokens_available.get("reactive", 0) if isinstance(tokens_available, dict) else tokens_available

        daily_records = convert_state_to_daily_records(
            state=state,
            episode_id=self.current_episode_id,
            tokens_available=tokens_count,
            session_id=self.session_id,
            episode_number=self.current_episode_number
        )

        # Write to database (all metadata already in daily_records)
        write_daily_data(
            conn=self.db_conn,
            records=daily_records
        )

    def finish_episode(self, metrics: Dict) -> None:
        """Finish episode and record final metrics.

        Args:
            metrics: Dict with final episode metrics
                - mean_or
                - final_or
                - total_mission_success_rate
                - total_flight_hours
                - total_inflight_failures
                - total_nmc_days
        """
        if not self.enabled or self.db_conn is None or self.current_episode_id is None:
            return

        # Update episode record with final metrics
        update_episode_metrics(
            conn=self.db_conn,
            episode_id=self.current_episode_id,
            metrics=metrics
        )

        if self.verbose:
            print(f"✅ Finished episode {self.current_episode_number} "
                  f"(mean_or={metrics.get('mean_or', 0):.2%})")

    def close(self) -> None:
        """Close database connection."""
        if self.enabled and self.db_conn is not None:
            close_database(self.db_conn)

            if self.verbose:
                print(f"📊 DataCollector closed: {self.db_path}")

            self.db_conn = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        """String representation."""
        status = "enabled" if self.enabled else "disabled"
        return (
            f"DataCollector(session={self.session_id}, "
            f"episode={self.current_episode_number}, "
            f"status={status})"
        )


class NoOpDataCollector:
    """No-op data collector for when collection is disabled.

    Provides same interface as DataCollector but does nothing.
    Useful for performance when detailed data collection is not needed.
    """

    def __init__(self, *args, **kwargs):
        """Initialize no-op collector."""
        pass

    def start_episode(self, episode_number: int, sim_days: int) -> int:
        """No-op start episode."""
        return -1

    def collect_daily_data(self, day: int, state: Dict, actions: Dict, tokens_available: Dict) -> None:
        """No-op collect daily data."""
        pass

    def finish_episode(self, metrics: Dict) -> None:
        """No-op finish episode."""
        pass

    def close(self) -> None:
        """No-op close."""
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass

    def __repr__(self) -> str:
        """String representation."""
        return "NoOpDataCollector()"


def create_data_collector(
    db_path: Optional[str] = None,
    session_id: int = 1,
    enabled: bool = True,
    verbose: bool = False
) -> DataCollector:
    """Factory function to create appropriate data collector.

    Args:
        db_path: Path to database file (None if collection disabled)
        session_id: Session identifier
        enabled: Enable data collection
        verbose: Enable detailed logging

    Returns:
        DataCollector or NoOpDataCollector based on enabled flag
    """
    if not enabled or db_path is None:
        return NoOpDataCollector()

    return DataCollector(
        db_path=db_path,
        session_id=session_id,
        enabled=enabled,
        verbose=verbose
    )
