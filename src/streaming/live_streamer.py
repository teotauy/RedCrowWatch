#!/usr/bin/env python3
"""
RedCrowWatch Live Streamer

Captures Reolink RTSP feed, runs real-time AI analysis,
draws overlays, and streams to YouTube via ffmpeg RTMP.
"""

import os
import sys
import time
import logging
import subprocess
import threading
import collections
from datetime import datetime, date
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Detection zone definitions (adjust coordinates to your camera angle) ────
ZONES = {
    'intersection': {'color': (0, 255, 255), 'pts': None},   # set in config
}

# Vehicle class IDs in YOLO COCO (car=2, motorcycle=3, bus=5, truck=7)
VEHICLE_CLASSES = {2: 'car', 3: 'moto', 5: 'bus', 7: 'truck'}


class StreamStats:
    """Thread-safe counters for the on-screen stats panel."""

    def __init__(self):
        self._lock = threading.Lock()
        self.reset_date = date.today()
        self.violations_today = 0
        self.horn_events_today = 0
        self.vehicles_seen_today = 0
        self.stream_start = time.time()

    def _check_day_rollover(self):
        if date.today() != self.reset_date:
            self.violations_today = 0
            self.horn_events_today = 0
            self.vehicles_seen_today = 0
            self.reset_date = date.today()

    def add_violation(self):
        with self._lock:
            self._check_day_rollover()
            self.violations_today += 1

    def add_horn(self):
        with self._lock:
            self._check_day_rollover()
            self.horn_events_today += 1

    def add_vehicle(self):
        with self._lock:
            self._check_day_rollover()
            self.vehicles_seen_today += 1

    def snapshot(self):
        with self._lock:
            self._check_day_rollover()
            uptime_s = int(time.time() - self.stream_start)
            h, m, s = uptime_s // 3600, (uptime_s % 3600) // 60, uptime_s % 60
            return {
                'violations': self.violations_today,
                'horns': self.horn_events_today,
                'vehicles': self.vehicles_seen_today,
                'uptime': f'{h:02d}:{m:02d}:{s:02d}',
            }


class LiveStreamer:
    """
    Full pipeline: Reolink RTSP → YOLO detection → overlay → ffmpeg → YouTube.

    Usage:
        streamer = LiveStreamer(config)
        streamer.run()
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.stats = StreamStats()
        self._model = None
        self._yolo_available = False
        self._load_yolo()

    # ── Model loading ────────────────────────────────────────────────────────

    def _load_yolo(self):
        try:
            from ultralytics import YOLO
            model_path = self.cfg.get('yolo_model', 'yolov8n.pt')
            self._model = YOLO(model_path)
            self._yolo_available = True
            logger.info(f"YOLO loaded: {model_path}")
        except Exception as e:
            logger.warning(f"YOLO not available, running without detection: {e}")

    # ── ffmpeg process ───────────────────────────────────────────────────────

    def _start_ffmpeg(self, width: int, height: int, fps: float):
        """Spawn ffmpeg reading raw BGR frames from stdin, outputting to RTMP."""
        rtmp_url = self.cfg['youtube_rtmp_url']
        cmd = [
            'ffmpeg',
            '-loglevel', 'warning',
            # input: raw video from Python via pipe
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}',
            '-r', str(fps),
            '-i', 'pipe:0',
            # audio: silence (swap for -i <rtsp_url> to pass camera audio)
            '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo',
            # video encode
            '-vcodec', 'libx264',
            '-preset', 'veryfast',
            '-tune', 'zerolatency',
            '-crf', '23',
            '-maxrate', '6000k',
            '-bufsize', '12000k',
            '-pix_fmt', 'yuv420p',
            '-g', str(int(fps * 2)),  # keyframe every 2 seconds
            # audio encode
            '-acodec', 'aac',
            '-ar', '44100',
            '-b:a', '128k',
            # output
            '-f', 'flv',
            rtmp_url,
        ]
        logger.info("Starting ffmpeg...")
        return subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def _start_ffmpeg_with_audio(self, width: int, height: int, fps: float, rtsp_url: str):
        """ffmpeg variant that mixes in the camera's live audio."""
        rtmp_url = self.cfg['youtube_rtmp_url']
        cmd = [
            'ffmpeg',
            '-loglevel', 'warning',
            # video: raw BGR from Python
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}',
            '-r', str(fps),
            '-i', 'pipe:0',
            # audio: direct from camera RTSP
            '-rtsp_transport', 'tcp',
            '-i', rtsp_url,
            # video encode
            '-vcodec', 'libx264',
            '-preset', 'veryfast',
            '-tune', 'zerolatency',
            '-crf', '23',
            '-maxrate', '6000k',
            '-bufsize', '12000k',
            '-pix_fmt', 'yuv420p',
            '-g', str(int(fps * 2)),
            # audio encode (from camera stream, index 1)
            '-map', '0:v',
            '-map', '1:a',
            '-acodec', 'aac',
            '-ar', '44100',
            '-b:a', '128k',
            # output
            '-f', 'flv',
            rtmp_url,
        ]
        logger.info("Starting ffmpeg with camera audio...")
        return subprocess.Popen(cmd, stdin=subprocess.PIPE)

    # ── Overlay drawing ──────────────────────────────────────────────────────

    def _draw_detections(self, frame, results):
        """Draw bounding boxes and labels from YOLO results."""
        if results is None:
            return
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue
            conf = float(box.conf[0])
            if conf < self.cfg.get('detection_confidence', 0.4):
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{VEHICLE_CLASSES[cls_id]} {conf:.0%}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 1, cv2.LINE_AA)

    def _draw_zones(self, frame):
        """Draw detection zone overlays."""
        zones = self.cfg.get('detection_zones', [])
        for zone in zones:
            pts = np.array(zone['coordinates'], dtype=np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 255))
            cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)
            cv2.polylines(frame, [pts], True, (0, 255, 255), 1)
            # zone label
            cx = int(pts[:, 0].mean())
            cy = int(pts[:, 1].mean())
            cv2.putText(frame, zone.get('name', ''), (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    def _draw_stats_panel(self, frame):
        """Draw stats panel in top-right corner."""
        snap = self.stats.snapshot()
        now = datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
        lines = [
            f"RedCrowWatch",
            f"{now}",
            f"Uptime:      {snap['uptime']}",
            f"Vehicles:    {snap['vehicles']}",
            f"Violations:  {snap['violations']}",
            f"Horn events: {snap['horns']}",
        ]

        h, w = frame.shape[:2]
        panel_w, panel_h = 260, len(lines) * 22 + 16
        x0 = w - panel_w - 10
        y0 = 10

        # semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        for i, line in enumerate(lines):
            color = (0, 200, 255) if i == 0 else (220, 220, 220)
            thickness = 2 if i == 0 else 1
            cv2.putText(frame, line, (x0 + 8, y0 + 18 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, thickness, cv2.LINE_AA)

    def _draw_watermark(self, frame):
        h, w = frame.shape[:2]
        cv2.putText(frame, 'github.com/RedCrowWatch',
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (180, 180, 180), 1, cv2.LINE_AA)

    # ── Main loop ────────────────────────────────────────────────────────────

    def run(self):
        rtsp_url = self.cfg['rtsp_url']
        stream_width = self.cfg.get('stream_width', 1280)
        stream_height = self.cfg.get('stream_height', 720)
        stream_fps = float(self.cfg.get('stream_fps', 25))
        detect_every = self.cfg.get('detect_every_n_frames', 3)
        pass_audio = self.cfg.get('pass_camera_audio', True)

        logger.info(f"Opening RTSP stream: {rtsp_url}")
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            logger.error("Failed to open RTSP stream. Check URL and camera connectivity.")
            return

        logger.info(f"Stream opened. Output: {stream_width}x{stream_height} @ {stream_fps}fps")

        if pass_audio:
            ffmpeg_proc = self._start_ffmpeg_with_audio(stream_width, stream_height, stream_fps, rtsp_url)
        else:
            ffmpeg_proc = self._start_ffmpeg(stream_width, stream_height, stream_fps)

        last_results = None
        frame_count = 0
        prev_vehicle_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Frame read failed — reconnecting in 3s...")
                    time.sleep(3)
                    cap.release()
                    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    continue

                # Resize to stream dimensions
                if frame.shape[1] != stream_width or frame.shape[0] != stream_height:
                    frame = cv2.resize(frame, (stream_width, stream_height))

                # Run detection every N frames
                if self._yolo_available and frame_count % detect_every == 0:
                    last_results = self._model(frame, verbose=False,
                                               classes=list(VEHICLE_CLASSES.keys()))
                    # Count new vehicles
                    if last_results and last_results[0].boxes is not None:
                        current_count = sum(
                            1 for b in last_results[0].boxes
                            if int(b.cls[0]) in VEHICLE_CLASSES
                            and float(b.conf[0]) >= self.cfg.get('detection_confidence', 0.4)
                        )
                        if current_count > prev_vehicle_count:
                            for _ in range(current_count - prev_vehicle_count):
                                self.stats.add_vehicle()
                        prev_vehicle_count = current_count

                # Draw overlays
                self._draw_zones(frame)
                self._draw_detections(frame, last_results)
                self._draw_stats_panel(frame)
                self._draw_watermark(frame)

                # Write raw frame bytes to ffmpeg stdin
                try:
                    ffmpeg_proc.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    logger.error("ffmpeg pipe broken — exiting.")
                    break

                frame_count += 1

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            cap.release()
            if ffmpeg_proc.stdin:
                ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()
            logger.info("Stream ended.")
