# Cloud Database Integration Plan

## Goal

Replace the local SQLite file with a hosted PostgreSQL database on Render so that:
- `stream.py` running on the local Mac writes violations directly to the cloud
- The Render dashboard reads from the same database in real time
- No manual sync, no committed database files, no delay

## Current Architecture

```
Local Mac:
  stream.py → data/redcrowwatch.db (SQLite file, local only)
  dashboard.py (local) → reads same SQLite file

Render (public):
  dashboard.py → reads a stale committed copy of data/redcrowwatch.db
  (no connection to local machine — never sees live violations)
```

## Target Architecture

```
Local Mac:
  stream.py → DATABASE_URL → Render PostgreSQL ──┐
                                                  ↓
Render:                                     PostgreSQL DB
  dashboard.py → DATABASE_URL ──────────────────┘
  (reads live violations, updates within seconds)
```

## Files to Change

### 1. `src/database.py` — Add PostgreSQL support

Replace the current SQLite-only implementation with one that detects
`DATABASE_URL` in the environment and uses PostgreSQL when present,
falling back to SQLite for local dev without the env var set.

**Key changes:**
- Add `import os`, `import psycopg2` (or use `sqlalchemy`)
- In `__init__`, check `os.environ.get('DATABASE_URL')`
- If set: connect via psycopg2 (PostgreSQL)
- If not set: connect via sqlite3 (local SQLite, existing behavior)
- All public methods (`log_violation`, `get_violations`, etc.) keep the same signatures

**SQLite → PostgreSQL query translations needed:**
```
SQLite:  datetime('now', '-' || ? || ' days')
Postgres: NOW() - interval '%s days'   (use %s placeholder, not ?)

SQLite:  strftime('%Y-%m-%d %H:00:00', timestamp)
Postgres: date_trunc('hour', timestamp)

SQLite:  INTEGER PRIMARY KEY AUTOINCREMENT
Postgres: SERIAL PRIMARY KEY  (in CREATE TABLE)

SQLite:  ? placeholder
Postgres: %s placeholder
```

**Recommended approach — dual-mode factory:**
```python
import os
import sqlite3
import threading
from pathlib import Path

DATABASE_URL = os.environ.get('DATABASE_URL')

class ViolationDB:
    def __init__(self, db_path=None):
        self._lock = threading.Lock()
        self._use_postgres = bool(DATABASE_URL)
        if self._use_postgres:
            import psycopg2
            self._pg_url = DATABASE_URL
        else:
            self._db_path = db_path or str(Path(__file__).parent.parent / "data" / "redcrowwatch.db")
        self._init_db()

    def _connect(self):
        if self._use_postgres:
            import psycopg2
            return psycopg2.connect(self._pg_url)
        else:
            return sqlite3.connect(self._db_path)

    def _placeholder(self):
        return '%s' if self._use_postgres else '?'
```

Then each query method uses `self._connect()` and `self._placeholder()`.

For the time-range queries, use a helper:
```python
def _days_ago_expr(self, col='timestamp'):
    if self._use_postgres:
        return f"{col} > NOW() - interval '%s days'"
    else:
        return f"{col} > datetime('now', '-' || ? || ' days')"
```

For the hourly grouping:
```python
def _hour_trunc(self, col='timestamp'):
    if self._use_postgres:
        return f"date_trunc('hour', {col})"
    else:
        return f"strftime('%Y-%m-%d %H:00:00', {col})"
```

### 2. `requirements.txt` — Add PostgreSQL driver

Add:
```
psycopg2-binary>=2.9.9
```

### 3. `render.yaml` — Add PostgreSQL database service

```yaml
databases:
  - name: redcrowwatch-db
    plan: free
    databaseName: redcrowwatch
    user: redcrowwatch

services:
  - type: web
    name: redcrowwatch-dashboard
    env: python
    plan: starter
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn --bind 0.0.0.0:$PORT --timeout 120 dashboard:app
    envVars:
      - key: PYTHON_VERSION
        value: "3.12"
      - key: DATABASE_URL
        fromDatabase:
          name: redcrowwatch-db
          property: connectionString

  - type: cron
    name: redcrowwatch-daily-seed
    env: python
    schedule: "0 6 * * *"
    buildCommand: pip install -r requirements.txt
    startCommand: python seed_demo_data.py --daily
    envVars:
      - key: DATABASE_URL
        fromDatabase:
          name: redcrowwatch-db
          property: connectionString
```

Note: The free PostgreSQL database on Render expires after 90 days.
Upgrade to paid ($7/month) if you need it permanently.

### 4. `.env` — Add DATABASE_URL for local streaming to cloud

When you want `stream.py` on your Mac to write to the Render database:

```bash
# Get this from Render dashboard → redcrowwatch-db → Connection → External URL
DATABASE_URL=postgresql://redcrowwatch:PASSWORD@HOST:PORT/redcrowwatch
```

Leave `DATABASE_URL` unset for local-only SQLite operation.

### 5. `.env.example` — Document the new variable

Add:
```bash
# PostgreSQL connection string (optional)
# If set, violations are written to the cloud database (visible on Render dashboard)
# Get from: Render Dashboard → redcrowwatch-db → Connection → External URL
# Leave blank to use local SQLite only
DATABASE_URL=
```

### 6. `seed_demo_data.py` — Already compatible

The seed script already uses `ViolationDB()`. Once `database.py` is updated,
seeding will automatically use PostgreSQL when `DATABASE_URL` is set.
No changes needed.

### 7. `dashboard.py` — Already compatible

The dashboard already uses `ViolationDB()`. No changes needed after
`database.py` is updated.

## Render Setup Steps (Manual — Do Once)

1. Go to https://dashboard.render.com
2. Click **New** → **PostgreSQL**
3. Name: `redcrowwatch-db`, Region: same as your web service
4. Plan: Free (90-day limit) or Starter ($7/mo)
5. Click **Create Database**
6. Wait for it to provision (~1 minute)
7. In your **redcrowwatch-dashboard** web service → **Environment**:
   - Add `DATABASE_URL` = the **Internal Database URL** from the Postgres service
8. Redeploy the web service
9. On first boot, `_init_db()` will create the tables automatically (no migrations needed)

## Local → Cloud Streaming Setup

1. In Render: go to `redcrowwatch-db` → **Connection** → copy **External Database URL**
2. Add to your local `.env`:
   ```
   DATABASE_URL=postgresql://...external_url...
   ```
3. Run `stream.py` as normal — violations write directly to Render's PostgreSQL
4. Refresh the Render dashboard — you should see live violations appear within seconds

## Testing

```bash
# Test PostgreSQL connection locally
DATABASE_URL=postgresql://... python3 -c "
from src.database import ViolationDB
db = ViolationDB()
db.log_violation('test_violation', zone='test_zone', confidence=0.99)
violations = db.get_violations(days=1)
print('✓ PostgreSQL connected,', len(violations), 'violations found')
"

# Seed demo data to cloud database
DATABASE_URL=postgresql://... python3 seed_demo_data.py
```

## What to Do When Going Live

1. Complete the steps above — Render dashboard now shows real violations
2. Remove (or comment out) `DATABASE_URL` from `.env` if you don't want local
   test runs polluting the cloud database
3. Remove the sample data disclaimer from `templates/dashboard.html` (the amber
   `demo-banner` div at the top)
4. Delete or disable the `redcrowwatch-daily-seed` cron job in Render
   (real violations are now coming in — fake seed data no longer needed)

## Rollback

If PostgreSQL causes issues:
- Remove `DATABASE_URL` from `.env` and Render environment variables
- Both `stream.py` and `dashboard.py` fall back to SQLite automatically
- No code changes needed
