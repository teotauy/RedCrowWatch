#!/usr/bin/env python3
"""
RedCrowWatch Live Stream

Connects Reolink E1 Pro → real-time AI analysis → YouTube RTMP.
Detects vehicles and flags illegal 53-foot tractor-trailers.

Usage:
    python3 stream.py            # stream to YouTube (key required)
    python3 stream.py --preview  # local preview window, no key needed

Environment variables (set in .env):
    REOLINK_IP            - Camera IP address (required)
    REOLINK_USER          - Camera username (default: admin)
    REOLINK_PASS          - Camera password (required)
    YOUTUBE_STREAM_KEY    - YouTube live stream key (required for streaming)
    YOLO_MODEL            - YOLO model file (default: yolov8n.pt)
    STREAM_WIDTH          - Output width (default: 1280)
    STREAM_HEIGHT         - Output height (default: 720)
    STREAM_FPS            - Framerate (default: 25)
    DETECT_EVERY          - Run YOLO every N frames (default: 3)
    PASS_AUDIO            - Pass camera audio: 1 or 0 (default: 1)
    SEMI_ASPECT_RATIO     - Min width/height ratio to flag truck as semi (default: 2.0)
    SEMI_MIN_WIDTH_PX     - Min pixel width to flag truck as semi (default: 220)
    MJPEG_PORT            - Port for local network MJPEG preview (default: 8080, 0=off)
                            Open http://<mac-ip>:8080/ on any phone on the same WiFi

  Red-light detection (camera cannot see the signal directly):
    SIGNAL_CYCLE_SEC      - Full signal cycle in seconds, e.g. 90
    SIGNAL_GREEN_SEC      - Green-phase duration for 10th Ave, e.g. 35
    SIGNAL_PHASE_OFFSET   - Epoch time (seconds) when a red phase last started;
                            run preview, press C exactly when light turns red,
                            copy the printed value here, then restart.

  Calibration workflow:
    1.  python3 stream.py --preview
    2.  Watch the intersection; press C the instant 10th Ave turns RED
    3.  Copy the printed SIGNAL_PHASE_OFFSET=... line into .env
    4.  Restart:  python3 stream.py --preview
    5.  Confirm HUD shows Signal: RED/GREEN in sync with the actual light
    6.  Red-light violations will now fire reliably
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(name)s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('redcrowwatch')

sys.path.insert(0, str(Path(__file__).parent / 'src'))

PREVIEW_MODE = '--preview' in sys.argv


def build_config() -> dict:
    camera_ip   = os.environ.get('REOLINK_IP', '')
    camera_user = os.environ.get('REOLINK_USER', 'admin')
    camera_pass = os.environ.get('REOLINK_PASS', '')
    yt_key      = os.environ.get('YOUTUBE_STREAM_KEY', '')

    missing = []
    if not camera_ip:
        missing.append('REOLINK_IP')
    if not camera_pass:
        missing.append('REOLINK_PASS')
    if not PREVIEW_MODE and not yt_key:
        missing.append('YOUTUBE_STREAM_KEY')

    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        logger.error("Set them in .env — see stream.py header for details.")
        sys.exit(1)

    rtsp_url = f"rtsp://{camera_user}:{camera_pass}@{camera_ip}:554/h264Preview_01_main"

    # Detection zones — load from zones.json if the zone editor has saved one,
    # otherwise fall back to the hardcoded defaults below.
    zones_file = Path(__file__).parent / 'zones.json'
    if zones_file.exists():
        import json
        with open(zones_file) as _zf:
            detection_zones = json.load(_zf)['zones']
        logger.info(f"Loaded {len(detection_zones)} zones from zones.json")
    else:
        # Tenth Ave & 19th St, Brooklyn defaults
        # Edit visually with:  python3 zones.py
        detection_zones = [
            {
                # Tenth Ave travel lanes (diagonal, approaching intersection)
                'name': 'tenth_ave',
                'coordinates': [[290, 470], [560, 460], [620, 680], [290, 690]],
            },
            {
                # Intersection box where Tenth Ave meets 19th St
                'name': 'intersection',
                'coordinates': [[560, 460], [980, 460], [980, 680], [620, 680]],
            },
            {
                # 19th St approach and crosswalk (bottom right)
                'name': '19th_st_crosswalk',
                'coordinates': [[880, 620], [1270, 620], [1270, 720], [880, 720]],
            },
            {
                # Green-painted protected bike lane on Tenth Ave
                'name': 'bike_lane',
                'coordinates': [[555, 460], [640, 458], [660, 680], [575, 682]],
            },
        ]

    return {
        'rtsp_url':             rtsp_url,
        'youtube_rtmp_url':     f'rtmp://a.rtmp.youtube.com/live2/{yt_key}',
        'yolo_model':           os.environ.get('YOLO_MODEL', 'yolov8n.pt'),
        'stream_width':         int(os.environ.get('STREAM_WIDTH', 1280)),
        'stream_height':        int(os.environ.get('STREAM_HEIGHT', 720)),
        'stream_fps':           float(os.environ.get('STREAM_FPS', 25)),
        'detect_every_n_frames': int(os.environ.get('DETECT_EVERY', 3)),
        'detection_confidence': 0.3,   # lowered for night IR footage
        # YOLO internal inference size — YOLO resizes the full-res frame to this before
        # running inference, then maps detections back to original coordinates automatically.
        # Lower = faster / less CPU buffering. Try 320 if still buffering, 640 for accuracy.
        'yolo_imgsz':           int(os.environ.get('YOLO_IMGSZ', 480)),
        'pass_camera_audio':    os.environ.get('PASS_AUDIO', '1') == '1',
        'detection_zones':      detection_zones,
        # Semi detection thresholds — tune if getting false positives/negatives:
        #   Box trucks (14-26ft):  aspect ~1.8-2.9, width ~150-320px at intersection
        #   53ft semi + cab (~70ft): aspect ~2.5-5.8, width ~360-700px at intersection
        #   Raise these if box trucks are still triggering; lower if real semis are missed.
        #   Set SEMI_DEBUG=1 to overlay aspect+width on every truck for easy measurement.
        'semi_aspect_ratio':    float(os.environ.get('SEMI_ASPECT_RATIO', 2.5)),
        'semi_min_width_px':    int(os.environ.get('SEMI_MIN_WIDTH_PX', 360)),
        'semi_debug':           os.environ.get('SEMI_DEBUG', '0') == '1',
        # Frames a grid cell must be absent before it unlocks (~5s at 17fps)
        'semi_gap_frames':      int(os.environ.get('SEMI_GAP_FRAMES', 85)),
        # Seconds after any semi count before the next one can fire.
        # A 53-footer takes ~5-8s to clear an intersection at city speed.
        'semi_cooldown_sec':    float(os.environ.get('SEMI_COOLDOWN_SEC', 8.0)),
        # Zone name to restrict semi detection to (empty string = any zone)
        # Run  python3 zones.py  to edit zones visually
        'semi_zone':            os.environ.get('SEMI_ZONE', 'semi_detection'),
        # ── Signal timing (red-light detection without seeing the light) ──────
        # Set SIGNAL_CYCLE_SEC + SIGNAL_GREEN_SEC, then calibrate with the C key
        # in preview mode to capture SIGNAL_PHASE_OFFSET.
        'signal_cycle_sec':    float(os.environ.get('SIGNAL_CYCLE_SEC',  0)),
        'signal_green_sec':    float(os.environ.get('SIGNAL_GREEN_SEC',  0)),
        'signal_phase_offset': float(os.environ.get('SIGNAL_PHASE_OFFSET', 0)),
        # Master switch — set RED_LIGHT_ENABLED=0 in .env to silence all violations
        'red_light_enabled':   os.environ.get('RED_LIGHT_ENABLED', '1') == '1',
        # Grace period at the start of each red phase before violations are counted.
        # Covers cars clearing the intersection on yellow / stopping at the line.
        'red_light_grace_sec': float(os.environ.get('RED_LIGHT_GRACE_SEC', 6.0)),
        # Motion gate — violations only fire when pixel motion is detected in the
        # intersection zone.  Stopped cars at a red light produce no motion diff.
        # Lower motion_thresh or motion_pct if real runners are being missed;
        # raise them if parked/slow cars still trigger false positives.
        'red_light_motion_thresh': int(os.environ.get('RED_LIGHT_MOTION_THRESH', 20)),
        'red_light_motion_pct':  float(os.environ.get('RED_LIGHT_MOTION_PCT',   0.04)),
        # Local MJPEG preview server — open http://<mac-ip>:<port>/ on any device
        'mjpeg_port':          int(os.environ.get('MJPEG_PORT', 8080)),
        # Siren / emergency vehicle cooldown — minimum seconds between siren events.
        # A single ambulance or fire truck passing through typically triggers for
        # 10-20 seconds; 15s prevents counting the same vehicle multiple times.
        'siren_cooldown_sec':  float(os.environ.get('SIREN_COOLDOWN_SEC', 15.0)),
        # Pedestrian counting — zone-gated, deduplicated with 120px grid cells.
        # PED_ZONE: restrict counting to a named zone (empty = any active zone).
        #           e.g. set PED_ZONE=19th_st_crosswalk to count crosswalk traffic only.
        # PED_GAP_FRAMES: frames a 120px cell must be empty before it re-counts (~3s).
        'ped_zone':           os.environ.get('PED_ZONE', ''),
        'ped_gap_frames':     int(os.environ.get('PED_GAP_FRAMES', 60)),
    }


if __name__ == '__main__':
    logger.info("=" * 55)
    if PREVIEW_MODE:
        logger.info("  RedCrowWatch  |  Preview Mode")
    else:
        logger.info("  RedCrowWatch  |  Live Stream Pipeline")
    logger.info("=" * 55)

    config = build_config()

    logger.info(f"Camera:  {config['rtsp_url'].split('@')[-1]}")
    logger.info(f"Output:  {config['stream_width']}x{config['stream_height']} @ {config['stream_fps']}fps")
    logger.info(f"YOLO:    {config['yolo_model']}")
    semi_zone_label = config['semi_zone'] if config['semi_zone'] else 'all zones'
    logger.info(f"Semi detection:  aspect ≥ {config['semi_aspect_ratio']}  width ≥ {config['semi_min_width_px']}px  zone: {semi_zone_label}")
    if config['signal_cycle_sec'] and config['signal_green_sec']:
        red_sec = config['signal_cycle_sec'] - config['signal_green_sec']
        calibrated = "calibrated" if config['signal_phase_offset'] else "NOT calibrated — press C in preview when light turns red"
        logger.info(f"Signal timing:   {config['signal_cycle_sec']:.0f}s cycle  "
                    f"red={red_sec:.0f}s  green={config['signal_green_sec']:.0f}s  "
                    f"phase offset: {calibrated}")
    else:
        logger.info("Signal timing:   not configured — red-light detection disabled")
        logger.info("                 Set SIGNAL_CYCLE_SEC + SIGNAL_GREEN_SEC in .env")
        logger.info("                 then press C in preview to calibrate phase offset")
    if PREVIEW_MODE:
        logger.info("Mode:    LOCAL PREVIEW (no YouTube)")
        logger.info("         q = quit   s = save snapshot")
    else:
        logger.info(f"YouTube: rtmp://a.rtmp.youtube.com/live2/****")
        logger.info(f"Audio:   {'camera pass-through' if config['pass_camera_audio'] else 'silent'}")
    logger.info("-" * 55)

    from streaming.live_streamer import LiveStreamer
    streamer = LiveStreamer(config)
    streamer.run(preview=PREVIEW_MODE)
