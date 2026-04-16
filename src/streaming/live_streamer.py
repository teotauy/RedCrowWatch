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

# Import database for logging violations
try:
    from database import db
except ImportError:
    db = None  # Graceful fallback if database not available

logger = logging.getLogger(__name__)


# Vehicle class IDs in YOLO COCO (car=2, motorcycle=3, bus=5, truck=7)
VEHICLE_CLASSES = {2: 'car', 3: 'moto', 5: 'bus', 7: 'truck'}

# Pedestrian class in YOLO COCO
PERSON_CLASS = 0

# Traffic light class in YOLO COCO
TRAFFIC_LIGHT_CLASS = 9

# Zone names that are reserved for system use and excluded from vehicle detection
RESERVED_ZONE_NAMES = {'traffic_light_roi', 'pedestrian_signal_roi'}

# Colors (BGR)
COLOR_NORMAL      = (0, 200, 0)      # green  — regular vehicles
COLOR_SEMI        = (0, 0, 255)      # red    — illegal semi
COLOR_REDLIGHT    = (0, 30, 220)     # red-orange — red light runner
COLOR_ZONE        = (0, 255, 255)    # cyan   — detection zones
COLOR_PANEL_BG    = (0, 0, 0)
COLOR_TITLE       = (0, 200, 255)    # amber
COLOR_SEMI_HUD    = (0, 80, 255)     # red-orange for semi count in HUD
COLOR_RL_HUD      = (0, 30, 220)     # red for red-light count in HUD
COLOR_TEXT        = (220, 220, 220)
COLOR_SIG_RED     = (0, 0, 220)      # signal RED  label
COLOR_SIG_GREEN   = (0, 200, 0)      # signal GREEN label
COLOR_SIG_UNKNOWN = (120, 120, 120)  # signal unknown
COLOR_EMERG_HUD   = (50, 200, 255)   # amber-gold — emergency vehicle count in HUD
COLOR_PERSON      = (255, 200,  50)  # sky-blue   — pedestrian bounding boxes
COLOR_PERSON_HUD  = (255, 200,  50)  # sky-blue   — pedestrian count in HUD


class MjpegServer:
    """
    Serve the annotated frame as an MJPEG stream over HTTP.

    Any device on the same local network can open:
        http://<mac-ip>:<port>/
    in a browser or VLC and see the live annotated feed.

    Uses no third-party dependencies — raw sockets only.
    Quality is throttled to ~10fps at JPEG 60% to keep LAN bandwidth low.
    """

    def __init__(self, port: int = 8080):
        self.port  = port
        self._lock = threading.Lock()
        self._jpg  = b''

    def push(self, frame):
        """Update the frame served to all connected clients."""
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if ok:
            with self._lock:
                self._jpg = buf.tobytes()

    def _serve_client(self, sock):
        try:
            sock.recv(4096)   # consume the HTTP request
            sock.sendall(
                b'HTTP/1.1 200 OK\r\n'
                b'Content-Type: multipart/x-mixed-replace;boundary=rcw\r\n'
                b'Cache-Control: no-cache\r\n'
                b'\r\n'
            )
            while True:
                with self._lock:
                    jpg = self._jpg
                if jpg:
                    hdr = (
                        f'--rcw\r\n'
                        f'Content-Type: image/jpeg\r\n'
                        f'Content-Length: {len(jpg)}\r\n\r\n'
                    ).encode()
                    sock.sendall(hdr + jpg + b'\r\n')
                time.sleep(0.1)   # ~10 fps to the remote viewer
        except Exception:
            pass
        finally:
            try:
                sock.close()
            except Exception:
                pass

    def start(self):
        import socket

        def _accept_loop():
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(('', self.port))
            srv.listen(10)
            try:
                local_ip = socket.gethostbyname(socket.gethostname())
            except Exception:
                local_ip = '<your-mac-ip>'
            logger.info(f"MJPEG preview → http://{local_ip}:{self.port}/  (open on phone/browser)")
            while True:
                try:
                    conn, _ = srv.accept()
                    threading.Thread(
                        target=self._serve_client, args=(conn,),
                        daemon=True, name='mjpeg-client',
                    ).start()
                except Exception:
                    break

        threading.Thread(
            target=_accept_loop, daemon=True, name='mjpeg-server'
        ).start()


class StreamStats:
    """Thread-safe counters for the on-screen stats panel."""

    def __init__(self):
        self._lock = threading.Lock()
        self.reset_date = date.today()
        self.violations_today = 0
        self.semis_today = 0
        self.redlights_today = 0
        self.horn_events_today = 0
        self.emergencies_today = 0
        self.vehicles_seen_today = 0
        self.pedestrians_today = 0
        self.stream_start = time.time()

    def _check_day_rollover(self):
        if date.today() != self.reset_date:
            self.violations_today = 0
            self.semis_today = 0
            self.redlights_today = 0
            self.horn_events_today = 0
            self.emergencies_today = 0
            self.vehicles_seen_today = 0
            self.pedestrians_today = 0
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
            if db:
                db.log_violation("illegally_sized_semi", vehicle_class="truck")

    def add_redlight(self):
        with self._lock:
            self._check_day_rollover()
            self.redlights_today += 1
            self.violations_today += 1
            if db:
                db.log_violation("red_light_runner")

    def add_horn(self):
        with self._lock:
            self._check_day_rollover()
            self.horn_events_today += 1
            if db:
                db.log_violation("horn_honk")

    def add_emergency(self):
        """Count a siren / emergency-vehicle pass.  Does NOT increment violations."""
        with self._lock:
            self._check_day_rollover()
            self.emergencies_today += 1
            if db:
                db.log_violation("siren_detected")

    def add_pedestrian(self):
        with self._lock:
            self._check_day_rollover()
            self.pedestrians_today += 1
            if db:
                db.log_violation("pedestrian_mid_cycle_stranding")

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
                'violations':   self.violations_today,
                'semis':        self.semis_today,
                'redlights':    self.redlights_today,
                'horns':        self.horn_events_today,
                'emergencies':  self.emergencies_today,
                'vehicles':     self.vehicles_seen_today,
                'pedestrians':  self.pedestrians_today,
                'uptime':       f'{h:02d}:{m:02d}:{s:02d}',
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
        # Semi deduplication:
        # Global cooldown prevents a single moving rig being counted multiple
        # times as it crosses cell boundaries.  Cell tracking handles the case
        # where a semi is STATIONARY (parked/stopped) — it won't recount until
        # it leaves AND the cooldown has elapsed.
        self._semi_cell_last_frame: dict = {}
        self._semi_counted_cells:   set  = set()
        self._last_semi_counted_time: float = 0.0   # wall-clock time of last count

        # Pedestrian deduplication — 120px grid cells, cleared after ped_gap_frames
        self._ped_cell_last_frame: dict = {}
        self._ped_counted_cells:   set  = set()
        # Red-light runner tracking
        self._light_state: str        = 'unknown'   # 'red' | 'green' | 'yellow' | 'unknown'
        self._prev_intersection_keys: set = set()   # vehicle grid-keys in intersection last cycle
        self._red_phase_start_time: float = 0.0     # wall-clock when light last turned red
        self._prev_intersection_gray                = None   # grayscale crop for motion diff
        self._intersection_motion: bool   = False   # True when vehicles are moving in zone
        # Grid keys (cx//80, y2//80) of vehicles confirmed as red-light runners this phase.
        # Populated by _check_redlight_runners; cleared when light transitions off red.
        # Used by _draw_detections so the "RED LIGHT" label only appears on actual runners.
        self._redlight_vehicle_keys: set  = set()
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
        """Return True if point (cx, cy) falls inside any vehicle detection zone.
        Reserved zones (e.g. traffic_light_roi) are excluded."""
        for zone in self.cfg.get('detection_zones', []):
            if zone.get('name') in RESERVED_ZONE_NAMES:
                continue
            if self._point_in_poly(cx, cy, zone['coordinates']):
                return True
        return False

    def _bbox_touches_zone(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """
        Return True if ANY of several sample points from the bounding box
        falls inside a detection zone.

        Using multiple points handles large/cropped vehicles (like a 53-foot
        semi that extends past the frame edge) whose centroid might land
        outside a zone even though the truck is clearly crossing it.
        """
        cx, cy  = (x1 + x2) // 2, (y1 + y2) // 2
        samples = [
            (cx, cy),                    # centroid
            (cx, y2),                    # bottom-center (wheels)
            (x1 + (x2 - x1) // 4, y2),  # bottom-left quarter
            (x1 + 3 * (x2 - x1) // 4, y2),  # bottom-right quarter
        ]
        return any(self._in_any_active_zone(px, py) for px, py in samples)

    def _semi_in_zone(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """
        Return True if the semi's bbox overlaps the designated semi-detection zone.

        If SEMI_ZONE is set (e.g. 'zone_7'), only that zone is checked.
        If SEMI_ZONE is empty, falls back to checking all active zones.
        Uses the same multi-point sampling as _bbox_touches_zone so that
        large rigs that extend past the frame edge are still caught.
        """
        semi_zone_name = self.cfg.get('semi_zone', '')
        if not semi_zone_name:
            return self._bbox_touches_zone(x1, y1, x2, y2)

        cx  = (x1 + x2) // 2
        samples = [
            (cx,                        (y1 + y2) // 2),  # centroid
            (cx,                        y2),               # bottom-center
            (x1 + (x2 - x1) // 4,      y2),               # bottom-left quarter
            (x1 + 3 * (x2 - x1) // 4,  y2),               # bottom-right quarter
        ]
        for zone in self.cfg.get('detection_zones', []):
            if zone.get('name') == semi_zone_name:
                return any(
                    self._point_in_poly(px, py, zone['coordinates'])
                    for px, py in samples
                )
        # Named zone not found — fall back to any zone
        return self._bbox_touches_zone(x1, y1, x2, y2)

    # ── Semi-truck classification ─────────────────────────────────────────────

    def _is_semi(self, cls_id: int, x1: int, y1: int, x2: int, y2: int) -> bool:
        """
        Return True if this detection looks like a 53-foot tractor-trailer.

        Heuristics (tune via .env):
          SEMI_ASPECT_RATIO  — minimum width/height ratio for class-7 trucks (default 3.0)
          SEMI_MIN_WIDTH_PX  — minimum bounding box width for class-7 trucks (default 400)

        Size reference (side-on view at this intersection):
          Box truck  14ft: aspect ~1.8,  width ~150px   ← should NOT trigger
          Box truck  26ft: aspect ~2.9,  width ~300px   ← should NOT trigger
          53ft semi+cab:   aspect ~3.0+, width ~400px+  ← SHOULD trigger

        Class 7 (truck): normal path + cropped-cab path for when only the cab is in frame.
        Class 5 (bus):   only flagged when the aspect ratio AND width are extreme,
                         which is the signature of a 53-footer that YOLO has
                         misclassified.  Regular city buses have a far lower
                         aspect ratio (~1.2-2.0) and narrower bounding boxes.
        """
        if cls_id not in (5, 7):    # bus or truck
            return False
        w = x2 - x1
        h = y2 - y1
        if h == 0:
            return False
        aspect = w / h

        if self.cfg.get('semi_debug', False):
            cls_name = 'truck' if cls_id == 7 else 'bus'
            logger.debug(f"[SEMI_DEBUG] {cls_name} cls={cls_id} w={w}px h={h}px aspect={aspect:.2f}")

        if cls_id == 7:   # ── truck class ─────────────────────────────────
            min_aspect = self.cfg.get('semi_aspect_ratio', 2.5)
            min_width  = self.cfg.get('semi_min_width_px', 360)
            # Normal path: wide aspect + minimum width
            if aspect >= min_aspect and w >= min_width:
                return True
            # Cropped-cab path: semi cab fills more than half the frame width.
            # Requires aspect >= 2.0 so a nearby-but-short box truck can't trigger
            # this path just by being close to the camera.
            if w >= 640 and aspect >= 2.0:
                return True

        elif cls_id == 5:   # ── bus class ────────────────────────────────
            # City buses (MTA, school) have aspect ~1.2–2.0 at this distance.
            # A 53-footer misclassified as "bus" would have extreme dimensions.
            # Require both a very high aspect ratio AND very wide bbox so
            # we never flag a normal transit bus as an illegal semi.
            bus_min_aspect = self.cfg.get('semi_bus_aspect_ratio', 3.5)
            bus_min_width  = self.cfg.get('semi_bus_min_width_px',  480)
            if aspect >= bus_min_aspect and w >= bus_min_width:
                return True

        return False

    # ── Traffic-light state ───────────────────────────────────────────────────

    def _color_sample_light(self, frame, x1: int, y1: int, x2: int, y2: int) -> str:
        """
        Sample top-third vs bottom-third of a region to infer red/green/unknown.

        Traffic light layout (standard vertical US signal):
          top third   → red lamp
          bottom third → green lamp

        Compares R-channel dominance (red lamp) vs G-channel dominance (green
        lamp).  Returns 'red', 'green', or 'unknown'.
        Works in colour mode; returns 'unknown' in IR/night mode (equal channels).
        """
        h = y2 - y1
        if h < 8:
            return 'unknown'
        third     = max(1, h // 3)
        red_roi   = frame[y1        : y1 + third, x1:x2]
        green_roi = frame[y2 - third: y2,          x1:x2]
        if red_roi.size == 0 or green_roi.size == 0:
            return 'unknown'
        # BGR: channel 2 = R, channel 1 = G
        r_score = float(red_roi[:, :, 2].mean())   - float(red_roi[:, :, 1].mean())
        g_score = float(green_roi[:, :, 1].mean()) - float(green_roi[:, :, 2].mean())
        if r_score > 25:
            return 'red'
        if g_score > 25:
            return 'green'
        return 'unknown'

    def _detect_pedestrian_signal(self, frame) -> str:
        """
        Infer vehicle signal state from a visible pedestrian WALK/DON'T WALK head.

        Draw a zone named 'pedestrian_signal_roi' (in zones.py) tightly around
        the pedestrian signal head that controls the CROSSWALK CROSSING TENTH AVE.
        The relationship is direct:
          WALK (white figure)       → pedestrians cross Tenth Ave → cars on RED
          DON'T WALK (orange hand)  → pedestrians stopped          → cars on GREEN

        The countdown digits are also orange, so they correctly map to GREEN
        (cars still moving while pedestrians count down).

        Detection approach: only bright pixels are considered (background is black).
        Bright white → all channels roughly equal.
        Bright orange → red channel dominates strongly over blue.

        Returns 'walk', 'dont_walk', or 'unknown'.
        """
        for zone in self.cfg.get('detection_zones', []):
            if zone.get('name') != 'pedestrian_signal_roi':
                continue
            pts = np.array(zone['coordinates'], np.int32)
            x1  = int(pts[:, 0].min())
            y1  = int(pts[:, 1].min())
            x2  = int(pts[:, 0].max())
            y2  = int(pts[:, 1].max())
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return 'unknown'

            # Ignore dark background — only examine lit pixels
            luminance   = roi.mean(axis=2)           # shape (H, W)
            bright_mask = luminance > 50
            if bright_mask.sum() < 10:
                return 'unknown'                     # signal too dark to read

            r_mean = float(roi[:, :, 2][bright_mask].mean())
            g_mean = float(roi[:, :, 1][bright_mask].mean())
            b_mean = float(roi[:, :, 0][bright_mask].mean())

            # Orange (DON'T WALK / countdown): R >> B
            # White  (WALK):                  R ≈ G ≈ B, all channels balanced
            orange_score  = r_mean - b_mean
            balance_score = max(
                abs(r_mean - g_mean),
                abs(g_mean - b_mean),
                abs(r_mean - b_mean),
            )

            if orange_score > 50 and r_mean > g_mean:
                return 'dont_walk'        # orange hand or countdown digits
            if balance_score < 45:
                return 'walk'             # white pedestrian figure

            return 'unknown'

        return 'unknown'   # zone not defined

    def _get_light_state_from_timing(self) -> str:
        """
        Compute the expected signal state from a fixed-cycle timing plan.

        NYC signals run on deterministic fixed-time plans.  Once you know the
        cycle length and when the red phase starts, you can calculate the state
        at any moment from the system clock alone — no camera view of the light
        required.

        Configure via .env:
          SIGNAL_CYCLE_SEC    — full cycle length, seconds (e.g. 90)
          SIGNAL_GREEN_SEC    — green duration for the monitored direction (e.g. 35)
          SIGNAL_PHASE_OFFSET — epoch time (seconds) when a red phase last started;
                                calibrate by pressing C in preview mode the instant
                                the light turns red for the 10th Ave approach.

        Returns 'red', 'green', or 'unknown' (when not configured).
        """
        cycle_sec  = float(self.cfg.get('signal_cycle_sec',  0))
        green_sec  = float(self.cfg.get('signal_green_sec',  0))
        offset_sec = float(self.cfg.get('signal_phase_offset', 0))

        if cycle_sec <= 0 or green_sec <= 0:
            return 'unknown'

        red_sec = cycle_sec - green_sec
        elapsed = (time.time() - offset_sec) % cycle_sec
        # Cycle layout:  [─── red ───][─── green ───]
        return 'red' if elapsed < red_sec else 'green'

    def _get_light_state(self, frame, results) -> str:
        """
        Infer traffic light state using four methods in priority order:

        1. Pedestrian signal ROI — draw a zone named 'pedestrian_signal_roi'
           (in zones.py) around the WALK/DON'T WALK head for the crosswalk
           CROSSING Tenth Ave.  WALK (white) = Tenth Ave RED;
           DON'T WALK / countdown (orange) = Tenth Ave GREEN.
           Most reliable method; works whenever ped signals are visible.

        2. YOLO class-9 detection — works when the camera can see the vehicle
           signal head directly.  Often unreliable at night / IR.

        3. Fixed-cycle timing plan — uses SIGNAL_CYCLE_SEC / SIGNAL_GREEN_SEC /
           SIGNAL_PHASE_OFFSET from .env.  Deterministic backup; calibrate by
           pressing C in preview when 10th Ave turns red.

        4. Manual traffic_light_roi zone — colour-sample a zone drawn directly
           over the signal head.  Only useful when that head is in frame.

        Falls back to the last known state when nothing produces a result.
        """
        # ── Method 1: pedestrian WALK/DON'T WALK signal ROI ──────────────────
        ped = self._detect_pedestrian_signal(frame)
        if ped == 'walk':
            return 'red'       # WALK crossing Tenth Ave → cars on red
        if ped == 'dont_walk':
            return 'green'     # DON'T WALK / countdown  → cars on green

        # ── Method 2: YOLO class-9 vehicle signal ────────────────────────────
        if results is not None:
            for box in results[0].boxes:
                if int(box.cls[0]) != TRAFFIC_LIGHT_CLASS:
                    continue
                if float(box.conf[0]) < 0.3:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if y2 - y1 < 12:
                    continue
                state = self._color_sample_light(frame, x1, y1, x2, y2)
                if state != 'unknown':
                    return state

        # ── Method 3: fixed-cycle timing plan ────────────────────────────────
        state = self._get_light_state_from_timing()
        if state != 'unknown':
            return state

        # ── Method 4: manual traffic_light_roi zone (optional) ───────────────
        for zone in self.cfg.get('detection_zones', []):
            if zone.get('name') == 'traffic_light_roi':
                pts = np.array(zone['coordinates'], np.int32)
                x1  = int(pts[:, 0].min())
                y1  = int(pts[:, 1].min())
                x2  = int(pts[:, 0].max())
                y2  = int(pts[:, 1].max())
                state = self._color_sample_light(frame, x1, y1, x2, y2)
                if state != 'unknown':
                    return state
                break

        return 'unknown'   # no method produced a result — don't hold a stale state

    def _get_intersection_keys(self, results) -> set:
        """
        Return the set of grid-keys for vehicles currently in the 'intersection' zone.
        Grid key = (bottom_center_x // 80, bottom_y // 80) — coarse enough that the
        same vehicle maps to the same cell across successive detection frames.
        """
        if results is None:
            return set()
        conf_thresh = self.cfg.get('detection_confidence', 0.4)
        keys = set()
        for box in results[0].boxes:
            if int(box.cls[0]) not in VEHICLE_CLASSES:
                continue
            if float(box.conf[0]) < conf_thresh:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            # Check specifically the intersection zone (vehicles roll through here on red)
            for zone in self.cfg.get('detection_zones', []):
                if zone.get('name') == 'intersection' and \
                        self._point_in_poly(cx, y2, zone['coordinates']):
                    keys.add((cx // 80, y2 // 80))
                    break
        return keys

    def _compute_intersection_motion(self, frame) -> bool:
        """
        Return True if there is meaningful pixel motion inside the 'intersection'
        zone compared to the previous call.  Uses grayscale frame differencing on
        the zone's bounding box — fast and dependency-free.

        Logic:
          • Stopped cars at a red light produce ~0 diff (no movement).
          • A vehicle running the light produces a large diff as it crosses.
          • Pedestrians and minor lighting flicker are filtered by the threshold
            and the minimum-pixel-ratio requirement.

        Tune via .env:
          RED_LIGHT_MOTION_THRESH  — per-pixel diff threshold (default: 20)
          RED_LIGHT_MOTION_PCT     — min fraction of zone pixels that must move
                                     before motion is declared (default: 0.04 = 4%)
        """
        # Find the intersection zone bounding box
        zone_pts = None
        for zone in self.cfg.get('detection_zones', []):
            if zone.get('name') == 'intersection':
                zone_pts = np.array(zone['coordinates'], np.int32)
                break
        if zone_pts is None:
            return True   # zone not defined — don't block detection

        x1 = max(0, int(zone_pts[:, 0].min()))
        y1 = max(0, int(zone_pts[:, 1].min()))
        x2 = int(zone_pts[:, 0].max())
        y2 = int(zone_pts[:, 1].max())

        curr_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

        if self._prev_intersection_gray is None or \
                curr_gray.shape != self._prev_intersection_gray.shape:
            self._prev_intersection_gray = curr_gray
            return False

        thresh   = int(self.cfg.get('red_light_motion_thresh', 20))
        min_pct  = float(self.cfg.get('red_light_motion_pct',   0.04))

        diff         = cv2.absdiff(curr_gray, self._prev_intersection_gray)
        self._prev_intersection_gray = curr_gray
        motion_ratio = float((diff > thresh).sum()) / max(diff.size, 1)
        return motion_ratio >= min_pct

    def _check_redlight_runners(self, results):
        """
        Detect vehicles that newly entered the intersection while the light is red.
        Calls stats.add_redlight() for each new entry and logs a warning.

        Three gates must all pass before a violation is counted:
          1. RED_LIGHT_ENABLED = 1  (kill switch — set 0 in .env to disable)
          2. Grace period — RED_LIGHT_GRACE_SEC seconds must have elapsed since
             the red phase started.  Covers cars clearing on yellow / stopping.
          3. Motion gate — significant pixel movement must be present in the
             intersection zone.  Stopped cars sitting at a red produce no diff;
             a runner crossing the intersection does.
        """
        # ── Kill switch ───────────────────────────────────────────────────────
        if not self.cfg.get('red_light_enabled', True):
            return

        current_keys = self._get_intersection_keys(results)
        new_entries  = current_keys - self._prev_intersection_keys
        self._prev_intersection_keys = current_keys

        if self._light_state != 'red' or not new_entries:
            return

        # ── Grace period ──────────────────────────────────────────────────────
        grace_sec = float(self.cfg.get('red_light_grace_sec', 6.0))
        if time.time() - self._red_phase_start_time < grace_sec:
            return

        # ── Motion gate ───────────────────────────────────────────────────────
        # A vehicle stopped at the light produces zero diff.
        # A runner crossing produces a large diff — require it.
        if not self._intersection_motion:
            return

        for key in new_entries:
            self.stats.add_redlight()
            self._redlight_vehicle_keys.add(key)   # mark for overlay
        logger.warning(
            f"RED-LIGHT VIOLATION: {len(new_entries)} vehicle(s) entered intersection on red"
        )

    # ── Semi-truck deduplication ──────────────────────────────────────────────

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
        Return True (and record the count) only when:
          1. The global cooldown has elapsed since the last semi was counted, AND
          2. This grid cell isn't already occupied.

        The global cooldown (default 8 s) is the primary gate — a 53-footer
        takes 5-8 s to clear the intersection, so any detection within that
        window is the same rig crossing multiple grid cells.

        The cell gate handles stationary semis: a parked rig holds its cell
        indefinitely and won't recount even after the cooldown expires, until
        it moves away and the cell is cleared by _update_semi_cells.
        """
        cooldown = self.cfg.get('semi_cooldown_sec', 8.0)
        now = time.time()
        if now - self._last_semi_counted_time < cooldown:
            return False   # same rig still crossing — skip
        if grid_key not in self._semi_counted_cells:
            self._semi_counted_cells.add(grid_key)
            self._last_semi_counted_time = now
            return True
        return False

    # ── Pedestrian cell tracking ──────────────────────────────────────────────

    def _update_ped_cells(self, frame_count: int, active_cells: set):
        """Expire pedestrian grid cells absent for >= ped_gap_frames."""
        gap = self.cfg.get('ped_gap_frames', 60)
        for cell in active_cells:
            self._ped_cell_last_frame[cell] = frame_count
        cleared = {
            cell for cell in self._ped_counted_cells
            if frame_count - self._ped_cell_last_frame.get(cell, 0) >= gap
        }
        self._ped_counted_cells -= cleared

    def _should_count_pedestrian(self, grid_key: tuple) -> bool:
        """Return True (and lock the cell) if this is a new pedestrian position."""
        if grid_key not in self._ped_counted_cells:
            self._ped_counted_cells.add(grid_key)
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
            '-rtsp_transport', 'tcp',
            '-use_wallclock_as_timestamps', '1',   # fix non-monotonic DTS from camera
            '-i', rtsp_url,
            '-vcodec', 'libx264', '-preset', 'veryfast', '-tune', 'zerolatency',
            '-crf', '23', '-maxrate', '6000k', '-bufsize', '12000k',
            '-pix_fmt', 'yuv420p', '-g', str(int(fps * 2)),
            '-map', '0:v', '-map', '1:a',
            '-acodec', 'aac', '-ar', '44100', '-b:a', '128k',
            '-f', 'flv', rtmp_url,
        ]
        logger.info("Starting ffmpeg with camera audio...")
        return subprocess.Popen(cmd, stdin=subprocess.PIPE)

    # ── Horn detection ────────────────────────────────────────────────────────

    def _start_audio_capture(self, rtsp_url: str):
        """Start a dedicated ffmpeg subprocess that outputs raw PCM audio from the RTSP stream."""
        cmd = [
            'ffmpeg', '-loglevel', 'error',
            '-rtsp_transport', 'tcp',
            '-i', rtsp_url,
            '-vn',             # audio only
            '-ar', '22050',    # match AudioAnalyzer sample rate
            '-ac', '1',        # mono
            '-f', 's16le',     # raw 16-bit signed LE PCM
            'pipe:1',
        ]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def _audio_horn_thread(self, rtsp_url: str, stop_event: threading.Event):
        """
        Background thread: reads 2-second PCM chunks from RTSP audio,
        runs AudioAnalyzer.detect_horn_honks(), and ticks stats.add_horn()
        with a cooldown so a single sustained blast counts once.
        """
        try:
            from analysis.audio_analyzer import AudioAnalyzer
        except ImportError:
            logger.warning("AudioAnalyzer / librosa not installed — horn detection disabled.")
            return

        analyzer  = AudioAnalyzer()
        # Lower the siren duration threshold for live 2-second chunks.
        # The default 2.0s requires the entire chunk to be siren; 0.5s is more realistic.
        analyzer.config['siren_detection']['duration_min'] = 0.5

        sr        = 22050
        chunk_sec = 2
        # 16-bit mono: 2 bytes per sample
        chunk_bytes      = sr * chunk_sec * 2
        cooldown_sec     = float(self.cfg.get('horn_cooldown_sec', 3.0))
        siren_cooldown   = float(self.cfg.get('siren_cooldown_sec', 15.0))
        last_horn        = 0.0
        last_siren       = 0.0

        proc = self._start_audio_capture(rtsp_url)
        logger.info("Audio analysis thread started (horn + siren, 2-second chunks).")

        while not stop_event.is_set():
            try:
                raw = proc.stdout.read(chunk_bytes)
            except Exception:
                break
            if not raw or len(raw) < chunk_bytes // 2:
                # stream ended or nearly empty
                break

            # Convert raw signed-16 PCM → float32 in [-1, 1]
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

            # ── Horn detection ────────────────────────────────────────────────
            events = analyzer.detect_horn_honks(audio, sr)
            if events:
                now = time.time()
                if now - last_horn >= cooldown_sec:
                    self.stats.add_horn()
                    last_horn = now
                    best = max(events, key=lambda e: e['confidence'])
                    logger.info(
                        f"Horn detected  conf={best['confidence']:.0%}  "
                        f"power={best['power_db']:.1f}dB"
                    )

            # ── Siren / emergency vehicle detection ───────────────────────────
            siren_events = analyzer.detect_sirens(audio, sr)
            if siren_events:
                now = time.time()
                if now - last_siren >= siren_cooldown:
                    self.stats.add_emergency()
                    last_siren = now
                    best = max(siren_events, key=lambda e: e['confidence'])
                    logger.info(
                        f"Siren detected  conf={best['confidence']:.0%}  "
                        f"power={best['power_db']:.1f}dB  (ambulance or fire truck)"
                    )

        try:
            proc.terminate()
        except Exception:
            pass
        logger.info("Horn detection thread stopped.")

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
            # For regular vehicles use bottom-center (wheels on ground);
            # for semis use full bbox sampling to handle cropped/huge rigs.
            in_zone = self._in_any_active_zone(cx, y2)

            # Skip vehicles entirely outside all active zones — no box, no label.
            # This keeps the display clean (no boxes on cars queued at far-away lights).
            # Semi detection still gets its own zone check inside the semi branch.
            if not in_zone and not self._is_semi(cls_id, x1, y1, x2, y2):
                continue

            # Only label a vehicle "RED LIGHT" if it has been explicitly flagged by
            # _check_redlight_runners (entered the intersection zone while light was
            # confirmed red, after the grace period, with motion present).
            # This prevents stopped-at-red cars from being labelled, and respects
            # the RED_LIGHT_ENABLED kill switch at the display layer too.
            veh_key = (cx // 80, y2 // 80)
            is_redlight_runner = (
                self.cfg.get('red_light_enabled', True)
                and not self._is_semi(cls_id, x1, y1, x2, y2)
                and veh_key in self._redlight_vehicle_keys
            )

            if self._is_semi(cls_id, x1, y1, x2, y2) and self._semi_in_zone(x1, y1, x2, y2):
                color     = COLOR_SEMI
                label     = f"53' SEMI - ILLEGAL  {conf:.0%}"
                thickness = 3
                # 320px grid cells: a 53-footer crossing the intersection
                # stays in ~1-2 cells instead of crossing 4+ at 120px,
                # which was causing moving trucks to be counted on every cell boundary.
                grid_key  = (x1 // 320, y1 // 320)
                active_semi_cells.add(grid_key)

                # Count if cell hasn't been occupied recently
                if self._should_count_semi(grid_key):
                    self.stats.add_semi()
                    logger.warning(
                        f"Illegal 53' semi at ({x1},{y1})-({x2},{y2}) "
                        f"w={x2-x1}px ratio={(x2-x1)/(y2-y1) if y2!=y1 else 0:.1f} conf={conf:.0%}"
                    )

                # Subtle red fill
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_SEMI, -1)
                cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            elif is_redlight_runner:
                color     = COLOR_REDLIGHT
                label     = f"RED LIGHT  {conf:.0%}"
                thickness = 3
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

            # SEMI_DEBUG: draw aspect + width inside the bounding box for tuning
            if self.cfg.get('semi_debug', False) and cls_id in (5, 7):
                w_px = x2 - x1
                h_px = y2 - y1
                asp  = w_px / h_px if h_px else 0
                dbg  = f"w={w_px} asp={asp:.2f}"
                cv2.putText(frame, dbg, (x1 + 4, y1 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

        # Expire cells that have been absent long enough to have cleared
        self._update_semi_cells(frame_count, active_semi_cells)

        # ── Pedestrian detection ──────────────────────────────────────────────
        ped_zone_name  = self.cfg.get('ped_zone', '')
        active_ped_cells = set()

        for box in results[0].boxes:
            if int(box.cls[0]) != PERSON_CLASS:
                continue
            conf = float(box.conf[0])
            if conf < self.cfg.get('detection_confidence', 0.4):
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2

            # Zone gate — restrict to configured pedestrian zone (or any active zone)
            if ped_zone_name:
                in_ped_zone = False
                for zone in self.cfg.get('detection_zones', []):
                    if zone.get('name') == ped_zone_name:
                        in_ped_zone = self._point_in_poly(cx, y2, zone['coordinates'])
                        break
            else:
                in_ped_zone = self._in_any_active_zone(cx, y2)

            if not in_ped_zone:
                continue

            # Draw box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PERSON, 2)
            label = f'person {conf:.0%}'
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 6, y1), COLOR_PERSON, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            # Count with 120px cell deduplication
            grid_key = (cx // 120, y2 // 120)
            active_ped_cells.add(grid_key)
            if self._should_count_pedestrian(grid_key):
                self.stats.add_pedestrian()

        self._update_ped_cells(frame_count, active_ped_cells)

    def _draw_zones(self, frame):
        zones = self.cfg.get('detection_zones', [])
        for zone in zones:
            name = zone.get('name', '')
            pts  = np.array(zone['coordinates'], dtype=np.int32)

            if name == 'traffic_light_roi':
                # Draw with the current vehicle signal colour
                sig_color = {
                    'red':     COLOR_SIG_RED,
                    'green':   COLOR_SIG_GREEN,
                    'yellow':  (0, 200, 200),
                    'unknown': COLOR_SIG_UNKNOWN,
                }.get(self._light_state, COLOR_SIG_UNKNOWN)
                cv2.polylines(frame, [pts], True, sig_color, 2)
                x1 = int(pts[:, 0].min())
                y1 = int(pts[:, 1].min())
                cv2.putText(frame, f'SIG:{self._light_state.upper()}',
                            (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, sig_color, 1, cv2.LINE_AA)

            elif name == 'pedestrian_signal_roi':
                # Sample live and draw with detected WALK/DON'T WALK color
                ped = self._detect_pedestrian_signal(frame)
                ped_color = {
                    'walk':      (255, 255, 255),   # white — WALK active
                    'dont_walk': (0, 100, 255),     # orange — DON'T WALK / countdown
                    'unknown':   COLOR_SIG_UNKNOWN,
                }.get(ped, COLOR_SIG_UNKNOWN)
                cv2.polylines(frame, [pts], True, ped_color, 2)
                x1 = int(pts[:, 0].min())
                y1 = int(pts[:, 1].min())
                label = {'walk': 'WALK', 'dont_walk': "DON'T WALK"}.get(ped, 'PED:?')
                cv2.putText(frame, label,
                            (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, ped_color, 1, cv2.LINE_AA)

            else:
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], COLOR_ZONE)
                cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)
                cv2.polylines(frame, [pts], True, COLOR_ZONE, 1)
                cx = int(pts[:, 0].mean())
                cy = int(pts[:, 1].mean())
                cv2.putText(frame, name, (cx - 30, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_ZONE, 1, cv2.LINE_AA)

    def _draw_stats_panel(self, frame, fps_actual: float = 0.0, preview: bool = False):
        snap = self.stats.snapshot()
        now  = datetime.now().strftime('%Y-%m-%d  %H:%M:%S')

        sig_color = {
            'red':     COLOR_SIG_RED,
            'green':   COLOR_SIG_GREEN,
            'yellow':  (0, 200, 200),
            'unknown': COLOR_SIG_UNKNOWN,
        }.get(self._light_state, COLOR_SIG_UNKNOWN)
        sig_label = self._light_state.upper()

        lines = [
            ('RedCrowWatch',                            COLOR_TITLE,    True),
            (f'{now}',                                  COLOR_TEXT,     False),
            (f'Uptime:      {snap["uptime"]}',          COLOR_TEXT,     False),
            (f'Signal:      {sig_label}',               sig_color,      self._light_state == 'red'),
            (f'Vehicles:    {snap["vehicles"]}',        COLOR_TEXT,     False),
            (f'Pedestrians: {snap["pedestrians"]}',     COLOR_PERSON_HUD, False),
            (f'Violations:  {snap["violations"]}',      COLOR_TEXT,     False),
            (f'53\' Semis:  {snap["semis"]}',           COLOR_SEMI_HUD, snap['semis'] > 0),
            (f'Red lights:  {snap["redlights"]}',       COLOR_RL_HUD,   snap['redlights'] > 0),
            (f'Horn events: {snap["horns"]}',           COLOR_TEXT,     False),
            (f'Emergency:   {snap["emergencies"]}',     COLOR_EMERG_HUD, snap['emergencies'] > 0),
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
        # Sub-stream URL for audio-only connections (ffmpeg + audio analyzer).
        # The Reolink only allows one connection to the main stream at a time;
        # cv2 holds that connection for video.  The sub stream carries the same
        # audio at lower video resolution — we discard the video (-vn) anyway.
        rtsp_sub_url  = rtsp_url.replace('h264Preview_01_main', 'h264Preview_01_sub')
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
                    stream_width, stream_height, stream_fps, rtsp_sub_url
                )
            else:
                ffmpeg_proc = self._start_ffmpeg(stream_width, stream_height, stream_fps)

        last_results   = None
        frame_count    = 0
        prev_veh_count = 0
        fps_timer      = time.time()
        fps_actual     = 0.0

        # Start MJPEG preview server (local network phone/browser viewing)
        mjpeg_port = self.cfg.get('mjpeg_port', 8080)
        mjpeg = None
        if mjpeg_port:
            mjpeg = MjpegServer(port=mjpeg_port)
            mjpeg.start()

        # Start horn detection in background thread (uses sub stream — see rtsp_sub_url above)
        stop_horn  = threading.Event()
        horn_thread = threading.Thread(
            target=self._audio_horn_thread,
            args=(rtsp_sub_url, stop_horn),
            daemon=True,
            name='horn-detector',
        )
        horn_thread.start()

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

                # Run YOLO every N frames.
                # imgsz controls YOLO's internal inference resolution — YOLO letterboxes
                # the full-res frame down to this size, runs inference, then maps bounding
                # boxes back to the original frame coordinates automatically.
                # Lower imgsz = faster inference, less load on the CPU/GPU.
                if self._yolo_available and frame_count % detect_every == 0:
                    yolo_imgsz = self.cfg.get('yolo_imgsz', 480)
                    last_results = self._model(
                        frame, verbose=False, imgsz=yolo_imgsz,
                        classes=list(VEHICLE_CLASSES.keys()) + [TRAFFIC_LIGHT_CLASS, PERSON_CLASS],
                    )
                    new_light_state = self._get_light_state(frame, last_results)
                    if new_light_state == 'red' and self._light_state != 'red':
                        self._red_phase_start_time = time.time()
                    if new_light_state != 'red' and self._light_state == 'red':
                        # Light just left red — clear per-vehicle runner flags so the
                        # "RED LIGHT" label disappears once the phase ends.
                        self._redlight_vehicle_keys.clear()
                    self._light_state = new_light_state
                    self._intersection_motion = self._compute_intersection_motion(frame)
                    self._check_redlight_runners(last_results)
                    if last_results and last_results[0].boxes is not None:
                        conf_thresh = self.cfg.get('detection_confidence', 0.4)
                        cur = sum(
                            1 for b in last_results[0].boxes
                            if int(b.cls[0]) in VEHICLE_CLASSES
                            and float(b.conf[0]) >= conf_thresh
                            and self._in_any_active_zone(
                                (int(b.xyxy[0][0]) + int(b.xyxy[0][2])) // 2,
                                int(b.xyxy[0][3]),  # bottom-center (wheels)
                            )
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

                # Push to local MJPEG server (phone / browser preview)
                if mjpeg is not None:
                    mjpeg.push(frame)

                # FPS tracking
                frame_count += 1
                if frame_count % 30 == 0:
                    now = time.time()
                    fps_actual = 30.0 / max(now - fps_timer, 0.001)
                    fps_timer = now

                if preview:
                    cv2.imshow('RedCrowWatch - Preview  (q=quit  s=snapshot  c=calibrate signal)', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("Preview closed.")
                        break
                    elif key == ord('s'):
                        fname = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        cv2.imwrite(fname, frame)
                        logger.info(f"Snapshot saved: {fname}")
                    elif key == ord('c'):
                        # Signal phase calibration:
                        # Press C the instant you observe the 10th Ave light turn RED.
                        # Copy the SIGNAL_PHASE_OFFSET value below into your .env file,
                        # then restart the stream.
                        offset = time.time()
                        logger.info("=" * 55)
                        logger.info("  SIGNAL CALIBRATION — 10th Ave light just turned RED")
                        logger.info(f"  Add this line to your .env:")
                        logger.info(f"  SIGNAL_PHASE_OFFSET={offset:.0f}")
                        logger.info("  Then restart:  python3 stream.py --preview")
                        logger.info("=" * 55)
                else:
                    try:
                        ffmpeg_proc.stdin.write(frame.tobytes())
                    except BrokenPipeError:
                        logger.error("ffmpeg pipe broken — exiting.")
                        break

        except KeyboardInterrupt:
            logger.info("Interrupted.")
        finally:
            stop_horn.set()
            cap.release()
            if preview:
                cv2.destroyAllWindows()
            elif ffmpeg_proc:
                if ffmpeg_proc.stdin:
                    ffmpeg_proc.stdin.close()
                ffmpeg_proc.wait()
            logger.info("Stream ended.")
