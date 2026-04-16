#!/usr/bin/env python3
"""Seed the database with demo violations for testing the dashboard.

Usage:
  python seed_demo_data.py          # Full reset: wipe and seed 7 days of data
  python seed_demo_data.py --daily  # Additive: append today's violations only (safe with real data)
"""

import sys
import sqlite3
import random
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent / "src"))
from database import ViolationDB

db_path = Path(__file__).parent / "data" / "redcrowwatch.db"
db_path.parent.mkdir(exist_ok=True)

ZONES = ["intersection", "tenth_ave", "semi_detection", "19th_st_crosswalk", "bike_lane"]
VIOLATION_TYPES = [
    "illegally_sized_semi",
    "red_light_runner",
    "horn_honk",
    "siren_detected",
    "pedestrian_mid_cycle_stranding",
]

# Realistic hourly weights: heavier during rush hours, light overnight
HOUR_WEIGHTS = [
    0.2, 0.1, 0.1, 0.1, 0.2, 0.4,   # 0-5am
    0.8, 2.5, 3.0, 2.0, 1.5, 1.5,   # 6-11am (morning rush peaks 8-9)
    1.5, 1.5, 1.5, 1.8, 2.8, 3.0,   # 12-5pm (afternoon rush peaks 4-5)
    2.5, 1.8, 1.2, 0.8, 0.5, 0.3,   # 6-11pm
]


def insert_violation(conn, ts, violation_type, zone, confidence):
    conn.execute(
        "INSERT INTO violations (timestamp, violation_type, zone, confidence) VALUES (?, ?, ?, ?)",
        (ts.isoformat(), violation_type, zone, confidence)
    )


def insert_vehicle(conn, ts, zone):
    conn.execute(
        "INSERT INTO vehicles (timestamp, count, zone, vehicle_class) VALUES (?, ?, ?, ?)",
        (ts.isoformat(), random.randint(5, 50), zone, random.choice(["car", "truck", "bus", "moto"]))
    )


def insert_pedestrian(conn, ts, zone):
    conn.execute(
        "INSERT INTO pedestrians (timestamp, count, zone) VALUES (?, ?, ?)",
        (ts.isoformat(), random.randint(1, 20), zone)
    )


def seed_day(conn, day_start, n_violations=20):
    """Seed a single day's worth of violations with realistic hour weighting."""
    for _ in range(n_violations):
        hour = random.choices(range(24), weights=HOUR_WEIGHTS, k=1)[0]
        minute = random.randint(0, 59)
        ts = day_start + timedelta(hours=hour, minutes=minute)
        insert_violation(conn, ts, random.choice(VIOLATION_TYPES), random.choice(ZONES), random.uniform(0.7, 0.99))

    # A handful of vehicle and ped counts per day
    for _ in range(14):
        hour = random.randint(6, 22)
        ts = day_start + timedelta(hours=hour, minutes=random.randint(0, 59))
        insert_vehicle(conn, ts, random.choice(ZONES))

    for _ in range(11):
        hour = random.randint(7, 21)
        ts = day_start + timedelta(hours=hour, minutes=random.randint(0, 59))
        insert_pedestrian(conn, ts, random.choice(["19th_st_crosswalk", "bike_lane"]))


daily_mode = "--daily" in sys.argv

db = ViolationDB(str(db_path))
conn = sqlite3.connect(str(db_path))

if daily_mode:
    # Additive: just add today. Safe to run while real data is accumulating.
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    seed_day(conn, today, n_violations=20)
    conn.commit()
    conn.close()
    print(f"✓ Added ~20 demo violations for {today.strftime('%A %b %d')}")
    print("  (--daily mode: existing data preserved)")

else:
    # Full reset: wipe everything and seed 7 complete days
    conn.execute("DELETE FROM violations")
    conn.execute("DELETE FROM vehicles")
    conn.execute("DELETE FROM pedestrians")
    conn.commit()
    print("Cleared existing data")

    now = datetime.now()
    total = 0
    for days_back in range(7, -1, -1):
        day_start = (now - timedelta(days=days_back)).replace(hour=0, minute=0, second=0, microsecond=0)
        n = random.randint(15, 25)
        seed_day(conn, day_start, n_violations=n)
        total += n

    conn.commit()
    conn.close()
    print(f"✓ Generated ~{total} demo violations across 8 days")

    # Summary
    violations = db.get_violations(7)
    counts = db.get_violation_counts(7)
    print("\n" + "=" * 50)
    print("Demo Data Summary (Last 7 Days)")
    print("=" * 50)
    print(f"Total violations: {len(violations)}")
    print("\nBreakdown by type:")
    for item in counts:
        print(f"  {item['violation_type']}: {item['count']}")
    print("\n✓ Ready to test dashboard!")
