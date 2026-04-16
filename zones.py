#!/usr/bin/env python3
"""
RedCrowWatch Zone Editor

Interactive tool for adjusting detection zones on a live camera frame.
Saves zones to zones.json which stream.py loads automatically.

Usage:
    python3 zones.py

Controls:
    Left-click near a corner  -> drag to move it
    Right-click / click body  -> select that zone
    Tab                       -> cycle to next zone
    n                         -> new zone (click corners, Enter to finish, Esc to cancel)
    d / Delete                -> delete selected zone
    r                         -> refresh frame from camera
    s                         -> save zones.json
    q                         -> quit  (warns if unsaved)

Reserved zone names (give your zone one of these names for special behaviour):
    pedestrian_signal_roi  — Draw tightly around the WALK/DON'T WALK signal
                             head for the crosswalk CROSSING Tenth Ave.
                             WALK (white) = Tenth Ave RED; DON'T WALK (orange)
                             = Tenth Ave GREEN.  This is the primary method for
                             red-light detection when the vehicle signal is not
                             visible from the camera.

    traffic_light_roi      — Colour-sample a vehicle signal head directly.
                             Only useful if the signal IS visible in frame.
"""

import json
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

import cv2
import numpy as np

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('zones')

ZONES_FILE = Path(__file__).parent / 'zones.json'

# Matches the defaults in stream.py — used when zones.json doesn't exist yet
DEFAULT_ZONES = [
    {'name': 'tenth_ave',         'coordinates': [[290, 470], [560, 460], [620, 680], [290, 690]]},
    {'name': 'intersection',      'coordinates': [[560, 460], [980, 460], [980, 680], [620, 680]]},
    {'name': 'bike_lane',         'coordinates': [[555, 460], [640, 458], [660, 680], [575, 682]]},
    {'name': '19th_st_crosswalk', 'coordinates': [[880, 620], [1270, 620], [1270, 720], [880, 720]]},
]

ZONE_COLORS = [
    (0, 255, 255),   # cyan
    (0, 255, 0),     # green
    (255, 180, 0),   # blue
    (255, 0, 255),   # magenta
    (0, 165, 255),   # orange
    (0, 220, 220),   # yellow-green
]

SNAP_RADIUS = 14   # pixels — click this close to a corner to grab it


# ── helpers ──────────────────────────────────────────────────────────────────

def _point_in_poly(px: int, py: int, poly: list) -> bool:
    n      = len(poly)
    inside = False
    j      = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def load_zones() -> list:
    if ZONES_FILE.exists():
        with open(ZONES_FILE) as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data['zones'])} zones from {ZONES_FILE}")
        return data['zones']
    logger.info("zones.json not found — starting from stream.py defaults")
    return [dict(z) for z in DEFAULT_ZONES]


def save_zones(zones: list):
    with open(ZONES_FILE, 'w') as f:
        json.dump({'zones': zones}, f, indent=2)
    logger.info(f"Saved {len(zones)} zones -> {ZONES_FILE}")


def grab_frame() -> np.ndarray:
    ip   = os.environ.get('REOLINK_IP', '')
    user = os.environ.get('REOLINK_USER', 'admin')
    pw   = os.environ.get('REOLINK_PASS', '')
    if not ip or not pw:
        logger.warning("REOLINK_IP / REOLINK_PASS not set — showing blank canvas")
        return np.zeros((720, 1280, 3), dtype=np.uint8)
    rtsp = f"rtsp://{user}:{pw}@{ip}:554/h264Preview_01_main"
    logger.info(f"Connecting to {rtsp.split('@')[-1]} ...")
    cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)
    # skip buffered frames so we get a fresh one
    for _ in range(8):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        logger.warning("Could not grab frame — showing blank canvas")
        return np.zeros((720, 1280, 3), dtype=np.uint8)
    logger.info("Frame grabbed.")
    return cv2.resize(frame, (1280, 720))


# ── editor ────────────────────────────────────────────────────────────────────

WIN = 'RedCrowWatch Zone Editor  |  s=save  n=new  d=delete  r=refresh  Tab=cycle  q=quit'


class ZoneEditor:

    def __init__(self, base_frame: np.ndarray, zones: list):
        self.base      = base_frame.copy()
        self.zones     = zones          # [{name, coordinates}, ...]
        self.sel       = 0
        self.drag_zone = -1
        self.drag_pt   = -1
        self.new_mode  = False
        self.new_pts: list = []
        self.dirty     = False

    # ── rendering ─────────────────────────────────────────────────────────────

    def _color(self, i: int):
        return ZONE_COLORS[i % len(ZONE_COLORS)]

    def _render(self) -> np.ndarray:
        frame = self.base.copy()

        for i, zone in enumerate(self.zones):
            pts   = np.array(zone['coordinates'], dtype=np.int32)
            color = self._color(i)
            selected = (i == self.sel)

            # semi-transparent fill
            ov = frame.copy()
            cv2.fillPoly(ov, [pts], color)
            cv2.addWeighted(ov, 0.18 if selected else 0.07, frame, 1 - (0.18 if selected else 0.07), 0, frame)

            # outline
            cv2.polylines(frame, [pts], True, color, 2 if selected else 1)

            # corners
            for px, py in zone['coordinates']:
                cv2.circle(frame, (px, py), 7 if selected else 4, color, -1)
                cv2.circle(frame, (px, py), 7 if selected else 4, (0, 0, 0), 1)

            # name label
            cx = int(pts[:, 0].mean())
            cy = int(pts[:, 1].mean())
            cv2.putText(frame, zone['name'], (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # new-zone mode overlay
        if self.new_mode:
            for px, py in self.new_pts:
                cv2.circle(frame, (px, py), 5, (255, 255, 255), -1)
            if len(self.new_pts) > 1:
                cv2.polylines(frame, [np.array(self.new_pts, np.int32)], False,
                              (255, 255, 255), 1)
            msg = f'NEW ZONE — click corners ({len(self.new_pts)} added)   Enter=finish  Esc=cancel'
            cv2.putText(frame, msg, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            name   = self.zones[self.sel]['name'] if self.zones else '(no zones)'
            status = f'Zone {self.sel + 1}/{len(self.zones)}: {name}'
            if self.dirty:
                status += '  [unsaved — press s]'
            cv2.putText(frame, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        return frame

    # ── mouse ─────────────────────────────────────────────────────────────────

    def _nearest_corner(self, x: int, y: int):
        """Return (zone_idx, pt_idx) of closest corner within SNAP_RADIUS, else (-1,-1)."""
        best, bz, bp = SNAP_RADIUS, -1, -1
        for zi, zone in enumerate(self.zones):
            for pi, (px, py) in enumerate(zone['coordinates']):
                d = ((px - x) ** 2 + (py - y) ** 2) ** 0.5
                if d < best:
                    best, bz, bp = d, zi, pi
        return bz, bp

    def mouse_cb(self, event, x, y, flags, param):
        if self.new_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.new_pts.append([x, y])
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            bz, bp = self._nearest_corner(x, y)
            if bz >= 0:
                self.drag_zone, self.drag_pt = bz, bp
                self.sel = bz
            else:
                for i, zone in enumerate(self.zones):
                    if _point_in_poly(x, y, zone['coordinates']):
                        self.sel = i
                        break

        elif event == cv2.EVENT_MOUSEMOVE and self.drag_zone >= 0:
            self.zones[self.drag_zone]['coordinates'][self.drag_pt] = [x, y]
            self.dirty = True

        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_zone = self.drag_pt = -1

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self) -> list:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, 1280, 720)
        cv2.setMouseCallback(WIN, self.mouse_cb)

        logger.info("Zone editor ready.  Drag corners to reshape.  s = save when done.")

        warned_unsaved = False
        while True:
            cv2.imshow(WIN, self._render())
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                if self.dirty and not warned_unsaved:
                    logger.warning("You have unsaved changes — press 's' to save or 'q' again to discard.")
                    warned_unsaved = True
                    continue
                break

            elif key == ord('s'):
                save_zones(self.zones)
                self.dirty        = False
                warned_unsaved    = False

            elif key == ord('r'):
                logger.info("Refreshing frame from camera...")
                self.base = grab_frame()

            elif key == 9:   # Tab
                if self.zones:
                    self.sel = (self.sel + 1) % len(self.zones)

            elif key == ord('n'):
                self.new_mode = True
                self.new_pts  = []

            elif key == 13 and self.new_mode:   # Enter — close new zone
                if len(self.new_pts) >= 3:
                    name = f'zone_{len(self.zones) + 1}'
                    self.zones.append({'name': name, 'coordinates': self.new_pts})
                    self.sel   = len(self.zones) - 1
                    self.dirty = True
                    logger.info(f"Zone '{name}' added — rename it in zones.json if you like.")
                self.new_mode = False
                self.new_pts  = []

            elif key == 27 and self.new_mode:   # Esc — cancel new zone
                self.new_mode = False
                self.new_pts  = []
                logger.info("New zone cancelled.")

            elif key in (ord('d'), 127) and not self.new_mode:   # d or Delete
                if self.zones:
                    removed = self.zones.pop(self.sel)
                    self.sel   = max(0, min(self.sel, len(self.zones) - 1))
                    self.dirty = True
                    logger.info(f"Deleted zone '{removed['name']}'")

            warned_unsaved = False if key != ord('q') else warned_unsaved

        cv2.destroyAllWindows()
        return self.zones


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    zones  = load_zones()
    frame  = grab_frame()
    editor = ZoneEditor(frame, zones)
    editor.run()
