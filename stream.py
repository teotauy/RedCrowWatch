#!/usr/bin/env python3
"""
RedCrowWatch Live Stream

Connects Reolink E1 Pro → real-time AI analysis → YouTube RTMP.

Usage:
    python3 stream.py

Environment variables (set in .env or shell):
    REOLINK_IP          - Camera IP address (required)
    REOLINK_USER        - Camera username (default: admin)
    REOLINK_PASS        - Camera password (required)
    YOUTUBE_STREAM_KEY  - YouTube live stream key (required)
    YOLO_MODEL          - Path to YOLO model file (default: yolov8n.pt)
    STREAM_WIDTH        - Output width in pixels (default: 1280)
    STREAM_HEIGHT       - Output height in pixels (default: 720)
    STREAM_FPS          - Output framerate (default: 25)
    DETECT_EVERY        - Run detection every N frames (default: 3)
    PASS_AUDIO          - Pass camera audio to stream: 1 or 0 (default: 1)
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
    if not yt_key:
        missing.append('YOUTUBE_STREAM_KEY')

    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        logger.error("Set them in your .env file or shell — see stream.py header for details.")
        sys.exit(1)

    rtsp_url = (
        f"rtsp://{camera_user}:{camera_pass}@{camera_ip}:554/h264Preview_01_main"
    )

    # Detection zones from config.yaml — rough defaults, tune to your camera angle
    detection_zones = [
        {
            'name': 'intersection',
            'coordinates': [[200, 150], [1080, 150], [1080, 570], [200, 570]],
        },
        {
            'name': 'crosswalk',
            'coordinates': [[200, 570], [800, 570], [800, 680], [200, 680]],
        },
        {
            'name': 'bike_lane',
            'coordinates': [[50, 100], [200, 100], [200, 400], [50, 400]],
        },
    ]

    return {
        'rtsp_url': rtsp_url,
        'youtube_rtmp_url': f'rtmp://a.rtmp.youtube.com/live2/{yt_key}',
        'yolo_model': os.environ.get('YOLO_MODEL', 'yolov8n.pt'),
        'stream_width':  int(os.environ.get('STREAM_WIDTH', 1280)),
        'stream_height': int(os.environ.get('STREAM_HEIGHT', 720)),
        'stream_fps':    float(os.environ.get('STREAM_FPS', 25)),
        'detect_every_n_frames': int(os.environ.get('DETECT_EVERY', 3)),
        'detection_confidence': 0.4,
        'pass_camera_audio': os.environ.get('PASS_AUDIO', '1') == '1',
        'detection_zones': detection_zones,
    }


if __name__ == '__main__':
    logger.info("=" * 55)
    logger.info("  RedCrowWatch  |  Live Stream Pipeline")
    logger.info("=" * 55)

    config = build_config()

    logger.info(f"Camera:  {config['rtsp_url'].split('@')[-1]}")   # hide credentials
    logger.info(f"Output:  {config['stream_width']}x{config['stream_height']} @ {config['stream_fps']}fps")
    logger.info(f"YouTube: rtmp://a.rtmp.youtube.com/live2/****")
    logger.info(f"YOLO:    {config['yolo_model']}")
    logger.info(f"Audio:   {'camera pass-through' if config['pass_camera_audio'] else 'silent'}")
    logger.info("-" * 55)

    from streaming.live_streamer import LiveStreamer
    streamer = LiveStreamer(config)
    streamer.run()
