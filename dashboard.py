"""RedCrowWatch Dashboard - Flask web interface for violation analytics."""

import json
import os
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from database import db

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["JSON_SORT_KEYS"] = False


@app.route("/")
def index():
    """Main dashboard page."""
    return render_template("dashboard.html")


@app.route("/api/stats")
def api_stats():
    """Get current violation statistics."""
    days = request.args.get("days", 7, type=int)

    violations = db.get_violations(days)
    counts = db.get_violation_counts(days)
    hourly = db.get_hourly_counts(days)
    zones = db.get_zone_counts(days)

    total = len(violations)
    by_type = {item["violation_type"]: item["count"] for item in counts}

    return jsonify(
        {
            "total_violations": total,
            "by_type": by_type,
            "hourly": hourly,
            "by_zone": zones,
            "violations": violations[-20:],  # Last 20 for timeline
        }
    )


@app.route("/api/violations")
def api_violations():
    """Get violations with optional filtering."""
    days = request.args.get("days", 7, type=int)
    violation_type = request.args.get("type", None)
    limit = request.args.get("limit", 100, type=int)

    violations = db.get_violations(days, violation_type)
    return jsonify(violations[-limit:])


@app.route("/api/export")
def api_export():
    """Export violations as CSV."""
    days = request.args.get("days", 7, type=int)
    csv_data = db.export_csv(days)

    return csv_data, 200, {"Content-Disposition": "attachment; filename=violations.csv"}


@app.route("/health")
def health():
    """Health check."""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # Railway sets PORT, otherwise use 5001 for local dev
    port = int(os.environ.get("PORT", os.environ.get("DASHBOARD_PORT", 5001)))
    print(f"Starting RedCrowWatch Dashboard on http://0.0.0.0:{port}")
    app.run(debug=False, host="0.0.0.0", port=port)
