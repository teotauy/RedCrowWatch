#!/usr/bin/env python3
"""
Live NYC Intersection Analysis from RTSP stream.

Uses NYCIntersectionAnalyzer on a live RTSP camera feed and saves
detected violations to a timestamped CSV in data/outputs.

RTSP URL resolution order:
1. --rtsp-url CLI argument
2. CAMERA_RTSP_URL environment variable
3. camera.rtsp_url value in config/config.yaml
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import yaml

from analysis.nyc_intersection_analyzer import NYCIntersectionAnalyzer


logger = logging.getLogger(__name__)


def resolve_rtsp_url(cli_url: str | None, config_path: str) -> str:
    """Resolve RTSP URL from CLI, env, or config file."""
    if cli_url:
        return cli_url

    env_url = os.environ.get("CAMERA_RTSP_URL")
    if env_url:
        return env_url

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    rtsp_url = (
        cfg.get("camera", {}).get("rtsp_url")
        if isinstance(cfg.get("camera"), dict)
        else None
    )

    if not rtsp_url or "your_camera_ip" in rtsp_url:
        raise SystemExit(
            "RTSP URL is not configured.\n"
            "Set it via one of:\n"
            "  1) --rtsp-url rtsp://user:pass@ip:554/...\n"
            "  2) export CAMERA_RTSP_URL=rtsp://user:pass@ip:554/...\n"
            "  3) camera.rtsp_url in config/config.yaml (avoid committing secrets)"
        )

    return rtsp_url


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live NYC intersection analysis from RTSP camera feed"
    )
    parser.add_argument(
        "--config",
        "-c",
        default="config/config.yaml",
        help="Path to configuration YAML (default: config/config.yaml)",
    )
    parser.add_argument(
        "--rtsp-url",
        help="RTSP URL for the camera. "
        "If omitted, falls back to CAMERA_RTSP_URL env var, then config.camera.rtsp_url",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="data/outputs",
        help="Directory to write violations CSV (default: data/outputs)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    rtsp_url = resolve_rtsp_url(args.rtsp_url, args.config)
    logger.info(f"Using RTSP source: {rtsp_url}")

    analyzer = NYCIntersectionAnalyzer(config_path=args.config)

    logger.info("Starting live NYC intersection analysis from RTSP stream…")
    try:
        violations = analyzer.analyze_video(rtsp_url)
    except KeyboardInterrupt:
        logger.info("Interrupted by user, stopping analysis.")
        # No partial results to save here because analyze_video builds
        # its own list; for now we just exit cleanly.
        return

    # Persist results if the stream ended naturally
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"nyc_rtsp_violations_{timestamp}.csv"

    analyzer.save_violations_to_csv(violations, str(csv_path))
    logger.info("Live RTSP analysis complete.")
    logger.info(f"Saved {len(violations)} violations to {csv_path}")


if __name__ == "__main__":
    main()

