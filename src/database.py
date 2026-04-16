"""SQLite database schema and utilities for RedCrowWatch violations tracking."""

import sqlite3
import threading
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "redcrowwatch.db"


class ViolationDB:
    """Thread-safe SQLite database for tracking violations."""

    def __init__(self, db_path: str = str(DB_PATH)):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    violation_type TEXT NOT NULL,
                    zone TEXT,
                    confidence REAL,
                    details TEXT,
                    vehicle_class TEXT,
                    bbox TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vehicles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    vehicle_class TEXT NOT NULL,
                    count INTEGER DEFAULT 1,
                    zone TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pedestrians (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    count INTEGER DEFAULT 1,
                    zone TEXT
                )
                """
            )
            conn.commit()

    def log_violation(
        self,
        violation_type: str,
        zone: str = None,
        confidence: float = None,
        vehicle_class: str = None,
        details: str = None,
    ):
        """Log a violation to the database."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO violations
                    (violation_type, zone, confidence, vehicle_class, details)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (violation_type, zone, confidence, vehicle_class, details),
                )
                conn.commit()

    def log_vehicle_count(self, count: int, zone: str = None):
        """Log vehicle count snapshot."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO vehicles (count, zone)
                    VALUES (?, ?)
                    """,
                    (count, zone),
                )
                conn.commit()

    def log_pedestrian_count(self, count: int, zone: str = None):
        """Log pedestrian count snapshot."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO pedestrians (count, zone)
                    VALUES (?, ?)
                    """,
                    (count, zone),
                )
                conn.commit()

    def get_violations(self, days: int = 7, violation_type: str = None):
        """Get violations from last N days."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            query = """
                SELECT * FROM violations
                WHERE timestamp > datetime('now', '-' || ? || ' days')
            """
            params = [days]

            if violation_type:
                query += " AND violation_type = ?"
                params.append(violation_type)

            query += " ORDER BY timestamp DESC"
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_violation_counts(self, days: int = 7):
        """Get violation counts by type for last N days."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT violation_type, COUNT(*) as count
                FROM violations
                WHERE timestamp > datetime('now', '-' || ? || ' days')
                GROUP BY violation_type
                ORDER BY count DESC
                """,
                (days,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_hourly_counts(self, days: int = 7):
        """Get violation counts by hour for last N days."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT
                    strftime('%Y-%m-%d %H:00:00', timestamp) as hour,
                    COUNT(*) as count
                FROM violations
                WHERE timestamp > datetime('now', '-' || ? || ' days')
                GROUP BY hour
                ORDER BY hour DESC
                """,
                (days,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_zone_counts(self, days: int = 7):
        """Get violation counts by zone for last N days."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT zone, COUNT(*) as count
                FROM violations
                WHERE timestamp > datetime('now', '-' || ? || ' days')
                AND zone IS NOT NULL
                GROUP BY zone
                ORDER BY count DESC
                """,
                (days,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def export_csv(self, days: int = 7):
        """Export violations as CSV."""
        import csv
        from io import StringIO

        violations = self.get_violations(days)
        if not violations:
            return ""

        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=violations[0].keys())
        writer.writeheader()
        writer.writerows(violations)
        return output.getvalue()


# Global instance
db = ViolationDB()
