# RedCrowWatch - Live Traffic Intersection Monitoring System

A real-time AI-powered traffic monitoring system for NYC intersections that streams directly to YouTube. Detects illegal 53-foot tractor-trailers, red-light runners, and counts pedestrian and vehicle traffic using computer vision and audio analysis.

## System Overview

RedCrowWatch combines YOLO v8 computer vision with live audio analysis to monitor traffic violations in real-time. The system streams to YouTube and maintains a local MJPEG preview server for network-based monitoring on phones/tablets.

**Live at:** Tenth Ave & 19th St, Brooklyn  
**Camera:** Reolink E1 Pro Outdoor (192.168.4.26)

### Key Features
- Live Streaming: Real-time RTMP stream to YouTube
- Illegal Semi Detection: Flags 53-foot tractor-trailers using aspect ratio + width heuristics
- Red-Light Detection: Detects vehicles running red lights (requires signal calibration)
- Audio Analysis: Horn honks and emergency siren detection
- Pedestrian Counting: Real-time pedestrian detection and counting in designated zones
- Live Statistics Panel: On-screen HUD showing violations, vehicle counts, emergencies
- MJPEG Preview: Local HTTP stream for phone/browser viewing on same WiFi
- Automated Scheduling: Cron-based auto-start/stop on weekday mornings and afternoons

## Quick Start

### Prerequisites
- Reolink E1 Pro camera (or compatible RTSP camera)
- Python 3.12+
- FFmpeg (for RTMP streaming and audio extraction)
- YouTube Live Stream Key (for YouTube streaming)

### Installation
```bash
# Clone the repository
git clone https://github.com/colbyblack/RedCrowWatch.git
cd RedCrowWatch

# Install dependencies
pip install -r requirements.txt

# Download YOLO model (first run only)
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Set up environment variables
cp .env.example .env
# Edit .env with your camera IP, password, and YouTube stream key
nano .env
```

### Configuration
Create/edit `.env` with:
```bash
REOLINK_IP=192.168.4.26           # Camera IP on your home network
REOLINK_USER=admin                 # Camera username
REOLINK_PASS=your_password         # Camera password
YOUTUBE_STREAM_KEY=your_key_here   # YouTube live stream key

# Detection tuning (optional)
SEMI_ASPECT_RATIO=2.5              # Min width/height ratio for 53-footer detection
SEMI_MIN_WIDTH_PX=360              # Min pixel width for semi detection
YOLO_IMGSZ=480                     # YOLO inference size (480 = faster, 640 = more accurate)
DETECT_EVERY=3                     # Run YOLO every N frames (3 = ~8fps inference at 25fps stream)

# Red-light detection (requires calibration with C key)
SIGNAL_CYCLE_SEC=90                # Full traffic signal cycle in seconds
SIGNAL_GREEN_SEC=35                # Green phase duration for your direction
SIGNAL_PHASE_OFFSET=0              # Set by pressing C in preview mode

# Pedestrian counting
PED_ZONE=19th_st_crosswalk         # Zone to restrict pedestrian counting to
PED_GAP_FRAMES=60                  # Frames before same pedestrian can be recounted

# Network/streaming
MJPEG_PORT=8080                    # Local preview server port (0 = disabled)
STREAM_WIDTH=1280
STREAM_HEIGHT=720
STREAM_FPS=25
```

### Running

**Stream to YouTube (production):**
```bash
python3 stream.py
```

**Local preview only (testing/calibration):**
```bash
python3 stream.py --preview
```

**Auto-start via cron (weekdays 7am-9am and 2:30pm-5:30pm):**
```bash
bash scripts/install_cron.sh
# Edit START_TIME/STOP_TIME in scripts/install_cron.sh first if needed
```

**Manual start/stop:**
```bash
bash scripts/start_redcrowwatch.sh    # Start streaming
bash scripts/stop_redcrowwatch.sh     # Stop gracefully
```

## Detection Features

### 1. Illegal 53-Foot Tractor-Trailers
- **Detection Method**: YOLO class-7 (truck) with size filtering
- **Thresholds**: Aspect ratio ≥ 2.5, width ≥ 360px (at intersection distance)
- **False Positive Avoidance**: Zone-gated (only counts in semi_detection zone), excludes parked vehicles outside zones
- **Size Reference**:
  - Box truck (14-26ft): aspect ~1.8-2.9, width ~150-320px → NOT flagged
  - 53ft semi+cab: aspect ~2.5-5.8, width ~360-700px → FLAGGED
- **Tuning**: Set `SEMI_DEBUG=1` in `.env` and run `--preview` to see live aspect/width measurements on all trucks

### 2. Red-Light Runners
- **Detection Method**: Signal cycle timing + motion detection + zone presence
- **Requirements**: 
  1. Calibrate signal timing: run preview, press **C** when light turns red, copy `SIGNAL_PHASE_OFFSET` value to `.env`
  2. Set `SIGNAL_CYCLE_SEC` and `SIGNAL_GREEN_SEC` in `.env`
  3. Set `RED_LIGHT_ENABLED=1` in `.env`
- **Motion Gate**: Only fires when pixel motion detected in intersection (stops false alarms on stationary cars)
- **Grace Period**: 6-second grace period at red phase start (vehicles clearing on yellow)

### 3. Horn Honks & Emergency Sirens
- **Detection Method**: Spectral analysis on audio stream
- **Horn Range**: 200-2000Hz, 0.5-3.0s duration
- **Siren Range**: 500-2000Hz, 0.5s+ duration
- **Cooldown**: 15 seconds between siren counts (prevents double-counting one passing vehicle)

### 4. Pedestrian Counting
- **Detection Method**: YOLO class-0 (person), zone-gated
- **Deduplication**: 120px grid cells, clears after 60 frames (~3 seconds at 20fps)
- **Zone Restriction**: Counts only in zone specified by `PED_ZONE` (e.g., crosswalks only)
- **Use Case**: Traffic impact analysis, pedestrian volumes during rush hour

### 5. Vehicle Counting
- **Detection Method**: All vehicle classes (car, motorcycle, bus, truck) in any active zone
- **Deduplication**: Per-frame, only counts new entries
- **Zone-Gated**: Excludes far-away queued cars at distant traffic lights

## Statistics Panel (Live HUD)

The on-screen overlay shows in real-time:
- **Timestamp** and uptime
- **Signal State** (RED / GREEN / UNKNOWN)
- **Vehicles** — total vehicles in zones
- **Pedestrians** — total pedestrians in zones
- **Violations** — combined count of all violation types
- **53' Semis** — illegal trucks (in red)
- **Red Lights** — red-light runners (in red)
- **Horn Events** — horn honks detected
- **Emergency** — sirens detected (in gold)

## Network Access

**YouTube Stream:**
- View live at: YouTube Studio → live stream URL

**Local MJPEG Preview (same WiFi as Mac):**
- Find your Mac's IP: `ifconfig | grep "inet " | grep -v 127.0.0.1`
- Open on phone: `http://<your-mac-ip>:8080/`

**Cron Logs:**
- View at: `logs/cron.log`

**Stream Logs:**
- View at: `logs/stream.log`

## Calibration

### Signal Timing Calibration (for red-light detection)
1. Run preview: `python3 stream.py --preview`
2. Watch the intersection, press **C** the instant the traffic light turns RED
3. Terminal will print `SIGNAL_PHASE_OFFSET=<epoch_seconds>`
4. Add to `.env`:
   ```bash
   SIGNAL_CYCLE_SEC=90    # Full red+green cycle time
   SIGNAL_GREEN_SEC=35    # Green phase duration
   SIGNAL_PHASE_OFFSET=<value from step 3>
   ```
5. Restart: `python3 stream.py --preview`
6. Verify HUD shows "Signal: RED" / "Signal: GREEN" in sync with actual light
7. Enable: `RED_LIGHT_ENABLED=1` in `.env`

### Zone Editing (detection areas)
1. Run: `python3 zones.py`
2. Interactive editor opens with live camera feed
3. Click to add/move zone points, drag to adjust
4. Press 'n' for new zone, 'd' to delete current zone
5. Close window to save

**Standard zones:**
- `tenth_ave` — Tenth Ave approach
- `intersection` — the main intersection box (red-light zone)
- `semi_detection` — where semis are flagged
- `19th_st_crosswalk` — pedestrian zone
- `bike_lane` — bike lane area

## Project Structure

```
RedCrowWatch/
├── stream.py                       # Main entry point for streaming
├── zones.py                        # Interactive zone editor
├── .env                            # Configuration (create from .env.example)
├── zones.json                      # Saved detection zones (auto-generated)
│
├── src/streaming/
│   ├── live_streamer.py           # Core streaming pipeline & YOLO detection
│   └── mjpeg_server.py            # Local MJPEG preview server
│
├── src/analysis/
│   └── audio_analyzer.py          # Horn/siren detection via spectral analysis
│
├── scripts/
│   ├── start_redcrowwatch.sh      # Idempotent start script
│   ├── stop_redcrowwatch.sh       # Graceful stop script
│   └── install_cron.sh            # Install cron schedule (weekday mornings/afternoons)
│
├── logs/
│   ├── stream.log                 # Live stream output and errors
│   └── cron.log                   # Cron execution logs
│
└── requirements.txt               # Python dependencies
```

## Technical Stack

- **Python 3.12** — Core language
- **YOLO v8 Nano** — Real-time object detection (ultralytics)
- **OpenCV 4.10+** — Video capture and frame processing
- **FFmpeg** — RTSP client and RTMP streaming backend
- **LibROSA 0.11+** — Audio spectral analysis for horn/siren detection
- **NumPy** — Numerical operations
- **YouTube RTMP** — Live streaming protocol

## Performance Tuning

### If buffering in the YouTube stream:
1. Lower `YOLO_IMGSZ` in `.env`: `YOLO_IMGSZ=320` (faster but less accurate)
2. Increase `DETECT_EVERY`: `DETECT_EVERY=4` (run YOLO less often)
3. Reduce `STREAM_FPS`: `STREAM_FPS=20` (lower framerate)

### If missing vehicle detections:
1. Raise `YOLO_IMGSZ`: `YOLO_IMGSZ=640` (slower, more accurate)
2. Lower `DETECT_EVERY`: `DETECT_EVERY=2` (run YOLO more often)

### If false positive semis (box trucks being flagged):
1. Raise `SEMI_ASPECT_RATIO`: `SEMI_ASPECT_RATIO=3.0`
2. Raise `SEMI_MIN_WIDTH_PX`: `SEMI_MIN_WIDTH_PX=400`

### If missing real semis:
1. Lower `SEMI_ASPECT_RATIO`: `SEMI_ASPECT_RATIO=2.3`
2. Lower `SEMI_MIN_WIDTH_PX`: `SEMI_MIN_WIDTH_PX=340`

**Tip:** Use `SEMI_DEBUG=1` in `.env` with `--preview` to see exact width/aspect measurements on every truck passing by.

## Security & Privacy

- **Local Processing**: All video analysis done locally; no cloud inference
- **Network**: RTSP stream from camera is encrypted (RTLS supports both HTTP and HTTPS)
- **Audio**: Only spectral features extracted, no audio stored or transmitted
- **YouTube**: Standard YouTube live stream terms apply
- **Logs**: Stream logs contain timestamps and violation data only, no video frames

## Use Cases

- **DOT Traffic Analysis**: Data for intersection redesign and safety improvements
- **Community Board Advocacy**: Evidence of traffic violations for policy discussions
- **Carrier Accountability**: Tracking which companies run red lights or use oversized vehicles
- **Safety Research**: Academic research on traffic patterns and violations

## Troubleshooting

### Camera not found
```bash
# Check camera is on same WiFi
ping 192.168.4.26

# Verify .env credentials match Reolink app
cat .env | grep REOLINK
```

### Stream won't start (ffmpeg error)
```bash
# Ensure full PATH is set (cron uses stripped PATH)
which ffmpeg
export PATH="/opt/homebrew/bin:$PATH"
python3 stream.py
```

### YouTube stream starts but no video appears
- Check YouTube Studio shows stream as "Live"
- Verify YouTube Stream Key in `.env` matches Studio settings
- Wait 10-30 seconds for stream to buffer on YouTube's end

### Red-light detection firing constantly
- Disable with `RED_LIGHT_ENABLED=0` while debugging
- Calibrate signal timing again with **C** key in preview
- Check motion gate is enabled (default: on)

### High CPU usage / buffering
- See "Performance Tuning" section above
- Reduce `STREAM_FPS` or increase `DETECT_EVERY`

## Future Roadmap

Near-term (6-12 months):
- Carrier Logo Recognition: Identify Amazon, FedEx, XPO, etc. on trailer sides to track company-specific violations
- Per-Carrier Violation Reporting: Dashboard showing violation counts broken down by shipping company
- Multi-Intersection Support: Deploy to additional NYC intersections (Broadway & Canal, Atlantic & Flatbush, etc.)

Medium-term (1-2 years):
- Web Dashboard: Centralized view of violations across multiple intersections with filtering and export
- Historical Analysis: Trend detection and reporting on violation patterns over time
- Citation Integration: Data formatted for NYPD/DOT enforcement workflows
- Traffic Signal Timing Analysis: Recommend signal cycle adjustments based on observed traffic patterns

Long-term (2+ years):
- Real-Time Alert System: Notifications to traffic management centers during peak violation periods
- Predictive Models: Identify high-risk times and conditions for traffic enforcement planning
- Equity Analysis: Ensure monitoring data is representative of actual street usage patterns
- Community Reporting: Public API for sharing anonymized violation data with community boards

## Support

For issues or questions:
1. Check `logs/stream.log` for error details
2. Run `python3 stream.py --preview` to test locally
3. Verify `.env` configuration matches your camera and network
4. Check GitHub issues or create a new one

---

**RedCrowWatch** is a volunteer traffic safety project designed to help communities collect data on dangerous vehicles and advocate for safer streets using real-time AI analysis.
