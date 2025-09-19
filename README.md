# RedCrowWatch - Traffic Intersection Monitoring System

A comprehensive AI-powered traffic monitoring system using Wyze cameras to detect and report traffic violations at intersections.

## Project Overview

This project aims to create an automated system that:
- Monitors traffic intersections using Wyze v4 cameras
- Uses computer vision to detect traffic violations (red light running, speeding, etc.)
- Generates automated reports and visualizations
- Publishes findings to social media platforms

## Phase Structure

### Phase 1: Manual Prototype
- Manual video collection and analysis
- Basic violation detection using OpenCV
- Manual social media posting
- **Goal**: Validate concept with 1 hour of footage

### Phase 1.5: Automated Data Pipeline
- Automated video collection via RTSP
- Continuous analysis and database storage
- Automated reporting with Grafana dashboards
- **Goal**: Fully automated daily reporting

### Phase 2: Live Stream Experience
- 24/7 public live stream
- Real-time violation detection
- Interactive public dashboard
- **Goal**: Public-facing traffic monitoring platform

## Current Status: Phase 1 Development

## Setup Instructions

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure your Wyze camera settings in `config.yaml`

3. Run the video analysis:
```bash
python src/analyze_video.py --input path/to/video.mp4
```

## Project Structure

```
RedCrowWatch/
├── src/                    # Source code
│   ├── analysis/          # Video analysis modules
│   ├── visualization/     # Data visualization
│   └── social/           # Social media integration
├── data/                  # Data storage
│   ├── videos/           # Raw video files
│   ├── processed/        # Processed data
│   └── outputs/          # Generated reports
├── config/               # Configuration files
└── tests/               # Test files
```

## Legal Considerations

⚠️ **Important**: This system monitors public intersections. Ensure compliance with:
- Local privacy laws
- Traffic monitoring regulations
- Data retention policies
- Social media platform terms of service

## Contributing

This is a personal project for traffic safety awareness and data collection.

