#!/usr/bin/env python3
"""
RedCrowWatch Live Stream

Connects Reolink E1 Pro → real-time AI analysis → YouTube RTMP.
Detects vehicles and flags illegal 53-foot tractor-trailers.

Usage:
    python3 stream.py            # stream to YouTube (key required)
    python3 stream.py --preview  # local preview window, no key needed

Environment variables (set in .env):
    REOLINK_IP          - Camera IP address (required)
    REOLINK_USER        - Camera username (default: admin)
    REOLINK_PASS        - Camera password (required)
    YOUTUBE_STREAM_KEY  - YouTube live stream key (required for streaming)
    YOLO_MODEL          - YOLO model file (default: yolov8n.pt)
    STREAM_WIDTH        - Output width (default: 1280)
    STREAM_HEIGHT       - Output height (default: 720)
    STREAM_FPS          - Framerate (default: 25)
    DETECT_EVERY        - Run YOLO every N frames (default: 3)
    PASS_AUDIO          - Pass camera audio: 1 or 0 (default: 1)
    SEMI_ASPECT_RATIO   - Min width/height ratio to flag as semi (default: 2.5)
    SEMI_MIN_WIDTH_PX   - Min pixel width to flag as semi (default: 280)
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

    # Detection zones — Tenth Ave & 19th St, Brooklyn
    # Satellite reference: parking lot is LEFT of camera view (excluded)
    # Tenth Ave runs diagonally; 19th St crosses from the right
    detection_zones = [
        {
            # Tenth Ave travel lanes (diagonal, approaching intersection)
            # Excludes the parking lot on the far left
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
            # Right side of Tenth Ave (camera right of travel lanes)
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
        'pass_camera_audio':    os.environ.get('PASS_AUDIO', '1') == '1',
        'detection_zones':      detection_zones,
        'semi_aspect_ratio':    float(os.environ.get('SEMI_ASPECT_RATIO', 2.5)),
        'semi_min_width_px':    int(os.environ.get('SEMI_MIN_WIDTH_PX', 280)),
        # Frames a cell must be absent before a new semi can be counted there (~1.5s at 17fps)
        'semi_gap_frames':      int(os.environ.get('SEMI_GAP_FRAMES', 25)),
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
    logger.info(f"Semi detection:  aspect ≥ {config['semi_aspect_ratio']}  width ≥ {config['semi_min_width_px']}px")
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
