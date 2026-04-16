#!/usr/bin/env python3
"""Seed the database with demo violations for testing the dashboard."""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import random

sys.path.insert(0, str(Path(__file__).parent / "src"))

from database import db

# Clear existing data
import sqlite3
db_path = Path(__file__).parent / "data" / "redcrowwatch.db"
if db_path.exists():
    conn = sqlite3.connect(str(db_path))
    conn.execute("DELETE FROM violations")
    conn.execute("DELETE FROM vehicles")
    conn.execute("DELETE FROM pedestrians")
    conn.commit()
    conn.close()
    print("Cleared existing data")

# Re-init database
db = db.__class__(str(db_path))
print("Initialized fresh database")

# Demo zones
zones = ["intersection", "tenth_ave", "semi_detection", "19th_st_crosswalk", "bike_lane"]
violation_types = [
    "illegally_sized_semi",
    "red_light_runner",
    "horn_honk",
    "siren_detected",
    "pedestrian_mid_cycle_stranding",  # People still in crosswalk when signal changes
]

# Generate 150 violations over the last 7 days
now = datetime.now()
for i in range(150):
    # Random timestamp in last 7 days
    days_back = random.randint(0, 7)
    hours_back = random.randint(0, 23)
    minutes_back = random.randint(0, 59)

    violation_time = now - timedelta(days=days_back, hours=hours_back, minutes=minutes_back)

    violation_type = random.choice(violation_types)
    zone = random.choice(zones)
    confidence = random.uniform(0.7, 0.99)

    # Insert directly to get specific timestamps
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        INSERT INTO violations (timestamp, violation_type, zone, confidence)
        VALUES (?, ?, ?, ?)
        """,
        (violation_time.isoformat(), violation_type, zone, confidence)
    )
    conn.commit()
    conn.close()

print("✓ Generated 150 demo violations")

# Generate vehicle count snapshots
vehicle_classes = ["car", "truck", "bus", "moto"]
for i in range(100):
    days_back = random.randint(0, 7)
    hours_back = random.randint(0, 23)

    count_time = now - timedelta(days=days_back, hours=hours_back)
    count = random.randint(5, 50)
    zone = random.choice(zones)
    vehicle_class = random.choice(vehicle_classes)

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO vehicles (timestamp, count, zone, vehicle_class) VALUES (?, ?, ?, ?)",
        (count_time.isoformat(), count, zone, vehicle_class)
    )
    conn.commit()
    conn.close()

print("✓ Generated 100 vehicle count snapshots")

# Generate pedestrian count snapshots
for i in range(80):
    days_back = random.randint(0, 7)
    hours_back = random.randint(0, 23)

    count_time = now - timedelta(days=days_back, hours=hours_back)
    count = random.randint(1, 20)
    zone = random.choice(["19th_st_crosswalk", "bike_lane"])

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO pedestrians (timestamp, count, zone) VALUES (?, ?, ?)",
        (count_time.isoformat(), count, zone)
    )
    conn.commit()
    conn.close()

print("✓ Generated 80 pedestrian count snapshots")

# Show summary
violations = db.get_violations(7)
counts = db.get_violation_counts(7)

print("\n" + "="*50)
print("Demo Data Summary (Last 7 Days)")
print("="*50)
print(f"Total violations: {len(violations)}")
print(f"\nBreakdown by type:")
for item in counts:
    print(f"  {item['violation_type']}: {item['count']}")

print("\n✓ Ready to test dashboard!")
print("\nRun:  python3 dashboard.py")
print("Then visit: http://localhost:5000")
