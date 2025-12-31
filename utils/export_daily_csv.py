"""CSV export utility for detailed_daily_operations.csv.

Exports daily data from SQLite database to CSV format matching R version.
"""

import sqlite3
import csv
from typing import Optional, List
from pathlib import Path


# Column order matching R version's detailed_daily_operations.csv
COLUMN_ORDER = [
    "daily_id",
    "episode_id",
    "day",
    "aircraft_id",
    "decision",
    "true_rul",
    "observed_rul",
    "total_flight_hours",
    "todays_flight_hours",
    "hours_since_minor_phase",
    "hours_since_major_phase",
    "status",
    "flew_today",
    "flight_failure_today",
    "maintenance_started",
    "maintenance_completed",
    "days_in_maintenance",
    "queue_position",
    "tokens_remaining_preventive",
    "tokens_remaining_minor_phase",
    "tokens_remaining_major_phase",
    "episode",
    "session_id",
]


def export_daily_operations_csv(
    db_path: str,
    output_path: str,
    episode_ids: Optional[List[int]] = None,
    session_id: Optional[int] = None,
) -> int:
    """Export daily_data table to CSV file.

    Args:
        db_path: Path to SQLite database
        output_path: Output CSV file path
        episode_ids: Optional list of episode IDs to filter (None = all episodes)
        session_id: Optional session ID to filter (None = all sessions)

    Returns:
        Number of rows exported
    """
    # Create output directory if needed
    output_dir = Path(output_path).parent
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Connect to database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    cursor = conn.cursor()

    # Build query with optional filters
    query = "SELECT * FROM daily_data"
    params = []
    filters = []

    if episode_ids is not None:
        placeholders = ",".join("?" * len(episode_ids))
        filters.append(f"episode_id IN ({placeholders})")
        params.extend(episode_ids)

    if session_id is not None:
        filters.append("session_id = ?")
        params.append(session_id)

    if filters:
        query += " WHERE " + " AND ".join(filters)

    # Order by episode, day, aircraft for consistent output
    query += " ORDER BY episode_id, day, aircraft_id"

    # Execute query
    cursor.execute(query, params)

    # Write to CSV
    row_count = 0
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=COLUMN_ORDER)
        writer.writeheader()

        for row in cursor:
            # Convert row to dictionary with explicit type casting for INTEGER columns
            row_dict = {}
            for col in COLUMN_ORDER:
                value = row[col]

                # Cast INTEGER columns to int (fixes binary data issue)
                # INTEGER columns: daily_id, episode_id, day, aircraft_id, flew_today,
                # flight_failure_today, days_in_maintenance, tokens_*, episode, session_id
                if col in ['daily_id', 'episode_id', 'day', 'aircraft_id',
                          'flew_today', 'flight_failure_today', 'days_in_maintenance',
                          'tokens_remaining_preventive', 'tokens_remaining_minor_phase',
                          'tokens_remaining_major_phase', 'episode', 'session_id']:
                    if value is None:
                        row_dict[col] = None
                    elif isinstance(value, bytes):
                        # Convert bytes to int using little-endian format
                        row_dict[col] = int.from_bytes(value, byteorder='little', signed=False)
                    else:
                        row_dict[col] = int(value)
                else:
                    row_dict[col] = value

            writer.writerow(row_dict)
            row_count += 1

    conn.close()

    return row_count


def export_episode_summary_csv(
    db_path: str,
    output_path: str,
    session_id: Optional[int] = None,
) -> int:
    """Export episodes table to CSV file.

    Args:
        db_path: Path to SQLite database
        output_path: Output CSV file path
        session_id: Optional session ID to filter (None = all sessions)

    Returns:
        Number of rows exported
    """
    # Create output directory if needed
    output_dir = Path(output_path).parent
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Connect to database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Build query with optional filter
    query = "SELECT * FROM episodes"
    params = []

    if session_id is not None:
        query += " WHERE session_id = ?"
        params.append(session_id)

    # Order by session, episode number
    query += " ORDER BY session_id, episode_number"

    # Execute query
    cursor.execute(query, params)

    # Get column names from first row
    rows = cursor.fetchall()
    if not rows:
        conn.close()
        # Create empty CSV with header
        with open(output_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([])
        return 0

    # Write to CSV
    column_names = rows[0].keys()
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=column_names)
        writer.writeheader()

        for row in rows:
            row_dict = {col: row[col] for col in column_names}
            writer.writerow(row_dict)

    conn.close()

    return len(rows)


def export_all_data(
    db_path: str,
    output_dir: str,
    session_id: Optional[int] = None,
) -> dict:
    """Export both daily_data and episodes tables to CSV files.

    Args:
        db_path: Path to SQLite database
        output_dir: Output directory for CSV files
        session_id: Optional session ID to filter (None = all sessions)

    Returns:
        Dict with export statistics
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Export daily operations
    daily_csv_path = Path(output_dir) / "detailed_daily_operations.csv"
    daily_count = export_daily_operations_csv(
        db_path=db_path,
        output_path=str(daily_csv_path),
        session_id=session_id
    )

    # Export episode summary
    episode_csv_path = Path(output_dir) / "episode_summary.csv"
    episode_count = export_episode_summary_csv(
        db_path=db_path,
        output_path=str(episode_csv_path),
        session_id=session_id
    )

    return {
        "daily_operations_rows": daily_count,
        "episode_summary_rows": episode_count,
        "daily_csv_path": str(daily_csv_path),
        "episode_csv_path": str(episode_csv_path),
    }


if __name__ == "__main__":
    """CLI usage example."""
    import argparse

    parser = argparse.ArgumentParser(description="Export simulation data to CSV")
    parser.add_argument("--db-path", required=True, help="Path to SQLite database")
    parser.add_argument("--output-dir", required=True, help="Output directory for CSV files")
    parser.add_argument("--session-id", type=int, help="Filter by session ID (optional)")

    args = parser.parse_args()

    # Export all data
    stats = export_all_data(
        db_path=args.db_path,
        output_dir=args.output_dir,
        session_id=args.session_id
    )

    print(f"Export complete!")
    print(f"  Daily operations: {stats['daily_operations_rows']} rows -> {stats['daily_csv_path']}")
    print(f"  Episode summary: {stats['episode_summary_rows']} rows -> {stats['episode_csv_path']}")
