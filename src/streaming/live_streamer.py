#!/usr/bin/env python3
"""
RedCrowWatch Live Streamer

Captures Reolink RTSP feed, runs real-time AI analysis,
draws overlays, and streams to YouTube via ffmpeg RTMP.

Detects standard vehicles + flags illegal 53-foot tractor-trailers.
"""

import os
import sys
import time
import logging
import subprocess
import threading
from datetime import datetime, date

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# Vehicle class IDs in YOLO COCO (car=2, motorcycle=3, bus=5, truck=7)
VEHICLE_CLASSES = {2: 'car', 3: 'moto', 5: 'bus', 7: 'truck'}

# Colors (BGR)
COLOR_NORMAL   = (0, 200, 0)      # green  — regular vehicles
COLOR_SEMI     = (0, 0, 255)      # red    — illegal semi
COLOR_ZONE     = (0, 255, 255)    # cyan   — detection zones
COLOR_PANEL_BG = (0, 0, 0)
COLOR_TITLE    = (0, 200, 255)    # amber
COLOR_SEMI_HUD = (0, 80, 255)     # red-orange for semi count in HUD
COLOR_TEXT     = (220, 220, 220)


class StreamStats:
    """Thread-safe counters for the on-screen stats panel."""

    def __init__(self):
        self._lock = threading.Lock()
        self.reset_date = date.today()
        self.violations_today = 0
        self.semis_today = 0
        self.horn_events_today = 0
        self.vehicles_seen_today = 0
        self.stream_start = time.time()

    def _check_day_rollover(self):
        if date.today() != self.reset_date:
            self.violations_today = 0
            self.semis_today = 0
            self.horn_events_today = 0
            self.vehicles_seen_today = 0
            self.reset_date = date.today()

    def add_violation(self):
        with self._lock:
            self._check_day_rollover()
            self.violations_today += 1

    def add_semi(self):
        with self._lock:
            self._check_day_rollover()
            self.semis_today += 1
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
            h = uptime_s // 3600
            m = (uptime_s % 3600) // 60
            s = uptime_s % 60
            return {
                'violations': self.violations_today,
                'semis':      self.semis_today,
                'horns':      self.horn_events_today,
                'vehicles':   self.vehicles_seen_today,
                'uptime':     f'{h:02d}:{m:02d}:{s:02d}',
            }


class LiveStreamer:
    """
    Full pipeline: Reolink RTSP → YOLO detection → overlay → ffmpeg → YouTube.

    Semi-truck detection:
      Any YOLO 'truck' detection whose bounding box aspect ratio (w/h) exceeds
      semi_aspect_ratio AND whose pixel width exceeds semi_min_width_px is
      flagged as an illegal 53-foot tractor-trailer.

    Usage:
        streamer = LiveStreamer(config)
        streamer.run()              # stream to YouTube
        streamer.run(preview=True)  # local preview window, no YouTube needed
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.stats = StreamStats()
        self._model = None
        self._yolo_available = False
        # Presence-based semi deduplication:
        # Only count a new semi once the previous one has fully cleared the cell.
        self._semi_cell_last_frame: dict = {}   # grid_key -> last frame_count seen
        self._semi_counted_cells:   set  = set() # cells occupied (need to clear before recount)
        self._load_yolo()

    # ── Model ────────────────────────────────────────────────────────────────

    def _load_yolo(self):
        try:
            from ultralytics import YOLO
            model_path = self.cfg.get('yolo_model', 'yolov8n.pt')
            self._model = YOLO(model_path)
            self._yolo_available = True
            logger.info(f"YOLO loaded: {model_path}")
        except Exception as e:
            logger.warning(f"YOLO not available — running without detection: {e}")

    # ── Zone membership ───────────────────────────────────────────────────────

    @staticmethod
    def _point_in_poly(px: int, py: int, poly: list) -> bool:
        """Ray-casting point-in-polygon test."""
        n = len(poly)
        inside = False
        x, y = px, py
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def _in_any_active_zone(self, cx: int, cy: int) -> bool:
        """Return True if centroid (cx, cy) falls inside any configured zone."""
        for zone in self.cfg.get('detection_zones', []):
            if self._point_in_poly(cx, cy, zone['coordinates']):
                return True
        return False

    # ── Semi-truck classification ─────────────────────────────────────────────

    def _is_semi(self, cls_id: int, x1: int, y1: int, x2: int, y2: int) -> bool:
        """
        Return True if this truck detection looks like a 53-foot tractor-trailer.

        Heuristics (tune via .env after watching the preview):
          SEMI_ASPECT_RATIO  — minimum width/height ratio (default 2.5)
          SEMI_MIN_WIDTH_PX  — minimum bounding box width in pixels (default 280)

        A 53-foot semi at typical NYC intersection distance is dramatically
        wider than it is tall and occupies a large chunk of the frame width.
        A box truck or pickup is much closer to square.
        """
        if cls_id != 7:  # only 'truck' class
            return False
        w = x2 - x1
        h = y2 - y1
        if h == 0:
            return False
        aspect = w / h
        min_aspect = self.cfg.get('semi_aspect_ratio', 2.5)
        min_width  = self.cfg.get('semi_min_width_px', 280)
        return aspect >= min_aspect and w >= min_width

    def _update_semi_cells(self, frame_count: int, active_cells: set):
        """
        Expire grid cells that have been absent for >= semi_gap_frames.
        Called once per detection cycle with the set of cells seen this frame.
        """
        gap = self.cfg.get('semi_gap_frames', 25)   # ~1.5s at 17fps
        # Update last-seen frame for currently active cells
        for cell in active_cells:
            self._semi_cell_last_frame[cell] = frame_count
        # Unblock cells that have been absent long enough to have fully cleared
        cleared = {
            cell for cell in self._semi_counted_cells
            if frame_count - self._semi_cell_last_frame.get(cell, 0) >= gap
        }
        self._semi_counted_cells -= cleared

    def _should_count_semi(self, grid_key: tuple) -> bool:
        """
        Return True (and mark cell occupied) only if this cell isn't already
        occupied by a semi that hasn't cleared yet.
        A parked semi stays in the cell → counted once, never recounted.
        A moving semi clears the cell → next semi can be counted.
        """
        if grid_key not in self._semi_counted_cells:
            self._semi_counted_cells.add(grid_key)
            return True
        return False

    # ── ffmpeg ────────────────────────────────────────────────────────────────

    def _start_ffmpeg(self, width: int, height: int, fps: float):
        rtmp_url = self.cfg['youtube_rtmp_url']
        cmd = [
            'ffmpeg', '-loglevel', 'warning',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}', '-r', str(fps), '-i', 'pipe:0',
            '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo',
            '-vcodec', 'libx264', '-preset', 'veryfast', '-tune', 'zerolatency',
            '-crf', '23', '-maxrate', '6000k', '-bufsize', '12000k',
            '-pix_fmt', 'yuv420p', '-g', str(int(fps * 2)),
            '-acodec', 'aac', '-ar', '44100', '-b:a', '128k',
            '-f', 'flv', rtmp_url,
        ]
        logger.info("Starting ffmpeg (silent audio)...")
        return subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def _start_ffmpeg_with_audio(self, width: int, height: int, fps: float, rtsp_url: str):
        rtmp_url = self.cfg['youtube_rtmp_url']
        cmd = [
            'ffmpeg', '-loglevel', 'warning',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}', '-r', str(fps), '-i', 'pipe:0',
            '-rtsp_transport', 'tcp', '-i', rtsp_url,
            '-vcodec', 'libx264', '-preset', 'veryfast', '-tune', 'zerolatency',
            '-crf', '23', '-maxrate', '6000k', '-bufsize', '12000k',
            '-pix_fmt', 'yuv420p', '-g', str(int(fps * 2)),
            '-map', '0:v', '-map', '1:a',
            '-acodec', 'aac', '-ar', '44100', '-b:a', '128k',
            '-f', 'flv', rtmp_url,
        ]
        logger.info("Starting ffmpeg with camera audio...")
        return subprocess.Popen(cmd, stdin=subprocess.PIPE)

    # ── Overlay drawing ───────────────────────────────────────────────────────

    def _draw_detections(self, frame, results, frame_count: int = 0):
        """Draw bounding boxes. Semis get red boxes and a violation label."""
        if results is None:
            return

        active_semi_cells = set()

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue
            conf = float(box.conf[0])
            if conf < self.cfg.get('detection_confidence', 0.4):
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            in_zone = self._in_any_active_zone(cx, cy)

            if self._is_semi(cls_id, x1, y1, x2, y2):
                color     = COLOR_SEMI
                label     = f"53' SEMI - ILLEGAL  {conf:.0%}"
                thickness = 3
                grid_key  = (x1 // 120, y1 // 120)
                active_semi_cells.add(grid_key)

                # Count only if in a zone and cell hasn't been occupied since last clearance
                if in_zone and self._should_count_semi(grid_key):
                    self.stats.add_semi()
                    logger.warning(f"Illegal 53' semi in zone at ({x1},{y1}) conf={conf:.0%}")

                # Subtle red fill
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_SEMI, -1)
                cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            else:
                color     = COLOR_NORMAL
                label     = f"{VEHICLE_CLASSES[cls_id]} {conf:.0%}"
                thickness = 2

            # Label with solid background
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        # Expire cells that have been absent long enough to have cleared
        self._update_semi_cells(frame_count, active_semi_cells)

    def _draw_zones(self, frame):
        zones = self.cfg.get('detection_zones', [])
        for zone in zones:
            pts = np.array(zone['coordinates'], dtype=np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], COLOR_ZONE)
            cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)
            cv2.polylines(frame, [pts], True, COLOR_ZONE, 1)
            cx = int(pts[:, 0].mean())
            cy = int(pts[:, 1].mean())
            cv2.putText(frame, zone.get('name', ''), (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_ZONE, 1, cv2.LINE_AA)

    def _draw_stats_panel(self, frame, fps_actual: float = 0.0, preview: bool = False):
        snap = self.stats.snapshot()
        now  = datetime.now().strftime('%Y-%m-%d  %H:%M:%S')

        lines = [
            ('RedCrowWatch',                       COLOR_TITLE,    True),
            (f'{now}',                             COLOR_TEXT,     False),
            (f'Uptime:      {snap["uptime"]}',     COLOR_TEXT,     False),
            (f'Vehicles:    {snap["vehicles"]}',   COLOR_TEXT,     False),
            (f'Violations:  {snap["violations"]}', COLOR_TEXT,     False),
            (f'53\' Semis:  {snap["semis"]}',      COLOR_SEMI_HUD, snap['semis'] > 0),
            (f'Horn events: {snap["horns"]}',      COLOR_TEXT,     False),
        ]
        if preview:
            lines.append((f'FPS: {fps_actual:.1f}  [PREVIEW]', (160, 160, 160), False))

        h, w    = frame.shape[:2]
        panel_w = 275
        panel_h = len(lines) * 22 + 16
        x0, y0  = w - panel_w - 10, 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), COLOR_PANEL_BG, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        for i, (text, color, bold) in enumerate(lines):
            cv2.putText(frame, text, (x0 + 8, y0 + 18 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color,
                        2 if bold else 1, cv2.LINE_AA)

    def _draw_watermark(self, frame):
        h, w = frame.shape[:2]
        cv2.putText(frame, 'RedCrowWatch | NYC Traffic Monitor',
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (180, 180, 180), 1, cv2.LINE_AA)

    # ── Core loop ─────────────────────────────────────────────────────────────

    def _open_capture(self, rtsp_url: str):
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def run(self, preview: bool = False):
        """
        Main entry point.
          preview=False  → stream to YouTube via ffmpeg (YOUTUBE_STREAM_KEY required)
          preview=True   → local cv2 window, no key needed
                           q = quit   s = save snapshot
        """
        rtsp_url      = self.cfg['rtsp_url']
        stream_width  = self.cfg.get('stream_width', 1280)
        stream_height = self.cfg.get('stream_height', 720)
        stream_fps    = float(self.cfg.get('stream_fps', 25))
        detect_every  = self.cfg.get('detect_every_n_frames', 3)
        pass_audio    = self.cfg.get('pass_camera_audio', True)

        logger.info(f"Opening RTSP: {rtsp_url.split('@')[-1]}")
        cap = self._open_capture(rtsp_url)

        if not cap.isOpened():
            logger.error("Cannot open RTSP stream — check camera IP, password, and network.")
            return

        logger.info(f"Stream open → {stream_width}x{stream_height} @ {stream_fps}fps")

        ffmpeg_proc = None
        if not preview:
            if pass_audio:
                ffmpeg_proc = self._start_ffmpeg_with_audio(
                    stream_width, stream_height, stream_fps, rtsp_url
                )
            else:
                ffmpeg_proc = self._start_ffmpeg(stream_width, stream_height, stream_fps)

        last_results   = None
        frame_count    = 0
        prev_veh_count = 0
        fps_timer      = time.time()
        fps_actual     = 0.0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Frame read failed — reconnecting in 3s...")
                    time.sleep(3)
                    cap.release()
                    cap = self._open_capture(rtsp_url)
                    continue

                if frame.shape[1] != stream_width or frame.shape[0] != stream_height:
                    frame = cv2.resize(frame, (stream_width, stream_height))

                # Run YOLO every N frames
                if self._yolo_available and frame_count % detect_every == 0:
                    last_results = self._model(
                        frame, verbose=False, classes=list(VEHICLE_CLASSES.keys())
                    )
                    if last_results and last_results[0].boxes is not None:
                        conf_thresh = self.cfg.get('detection_confidence', 0.4)
                        cur = sum(
                            1 for b in last_results[0].boxes
                            if int(b.cls[0]) in VEHICLE_CLASSES
                            and float(b.conf[0]) >= conf_thresh
                        )
                        if cur > prev_veh_count:
                            for _ in range(cur - prev_veh_count):
                                self.stats.add_vehicle()
                        prev_veh_count = cur

                # Draw overlays
                self._draw_zones(frame)
                self._draw_detections(frame, last_results, frame_count)
                self._draw_stats_panel(frame, fps_actual, preview)
                self._draw_watermark(frame)

                # FPS tracking
                frame_count += 1
                if frame_count % 30 == 0:
                    now = time.time()
                    fps_actual = 30.0 / max(now - fps_timer, 0.001)
                    fps_timer = now

                if preview:
                    cv2.imshow('RedCrowWatch - Preview  (q=quit  s=snapshot)', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("Preview closed.")
                        break
                    elif key == ord('s'):
                        fname = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        cv2.imwrite(fname, frame)
                        logger.info(f"Snapshot saved: {fname}")
                else:
                    try:
                        ffmpeg_proc.stdin.write(frame.tobytes())
                    except BrokenPipeError:
                        logger.error("ffmpeg pipe broken — exiting.")
                        break

        except KeyboardInterrupt:
            logger.info("Interrupted.")
        finally:
            cap.release()
            if preview:
                cv2.destroyAllWindows()
            elif ffmpeg_proc:
                if ffmpeg_proc.stdin:
                    ffmpeg_proc.stdin.close()
                ffmpeg_proc.wait()
            logger.info("Stream ended.")
