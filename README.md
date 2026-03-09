# RedCrowWatch - AI-Powered Traffic Intersection Monitoring System

A comprehensive, production-ready AI system that combines computer vision, audio analysis, and data visualization to monitor traffic intersections and detect violations in real-time.

## 🚦 System Overview

RedCrowWatch is a full-stack AI-powered traffic monitoring system designed specifically for NYC intersections. It combines cutting-edge computer vision with advanced audio processing to detect traffic violations, safety concerns, and generate actionable insights for traffic safety improvement.

### Key Features
- **🎥 Multi-Modal AI Analysis**: Computer vision + audio processing + data correlation
- **🌐 Modern Web Interface**: Drag-and-drop video upload with real-time progress tracking
- **📊 Comprehensive Dashboards**: Interactive charts, heatmaps, and detailed analytics
- **🔧 Camera Calibration**: Web-based and command-line calibration tools
- **📱 Social Media Integration**: Automated Twitter posting with traffic safety insights
- **🐳 Production Ready**: Docker containerization with cloud deployment support

## 🏗️ Architecture & Components

### Core System Architecture
- **Multi-Modal Analysis Engine**: Combines video and audio processing for comprehensive traffic monitoring
- **Modular Design**: Separate analyzers for video, audio, and integrated analysis
- **Web Interface**: Modern Flask-based web application with Bootstrap 5 and Chart.js
- **Deployment Ready**: Docker containerization with Railway, Render, and VPS support

### Phase Structure
- **Phase 1**: Manual prototype with basic violation detection ✅ **COMPLETED**
- **Phase 1.5**: Automated data pipeline with continuous analysis 🔄 **IN PROGRESS**
- **Phase 2**: Live stream experience with real-time monitoring 📋 **PLANNED**

## 🎯 Core Functionality

### 1. Video Analysis Engine
- **YOLO v8 Integration**: Real-time object detection using ultralytics YOLO
- **Vehicle Classification**: Cars, trucks, buses, motorcycles, bicycles
- **Multi-Zone Detection**: 6 specialized detection zones for NYC intersection
- **Traffic Signal Cycle Management**: 88-second cycle with 3 phases

#### Violation Detection Types
- **Red Light Running**: Uses traffic signal cycle logic with NYC right-on-red allowance
- **Speeding**: Speed calculation with NYC-specific thresholds (25 mph limit)
- **Wrong-Way Driving**: Direction-based violation detection
- **Bike Lane Violations**: Non-bicycle vehicles in bike lanes
- **Pedestrian Violations**: Vehicles blocking pedestrian crossings
- **Bus Lane Violations**: Non-bus vehicles in bus lanes
- **Expressway Offramp Violations**: Specialized offramp speed monitoring

### 2. Audio Analysis Engine
- **Multi-Format Support**: MP4, AVI, MOV, MKV, WMV with ffmpeg fallback
- **Real-Time Analysis**: 22.05kHz sample rate with configurable parameters
- **Spectral Analysis**: FFT-based frequency analysis for event detection

#### Audio Event Detection
- **Horn Honks**: 200-2000Hz frequency range, 0.1-3.0s duration
- **Emergency Sirens**: 500-2000Hz frequency range, 2.0s+ duration
- **Brake Squeals**: 2000-8000Hz frequency range, 0.05-0.5s duration
- **Audio-Video Correlation**: Temporal correlation with 2-second time window

### 3. Web Interface
- **Modern Web Application**: Flask framework with Bootstrap 5 and Chart.js
- **Drag & Drop Upload**: Support for multiple formats up to 500MB
- **Real-Time Progress**: Live analysis progress with visual indicators
- **Results Dashboard**: Comprehensive results with charts, tables, and visualizations
- **Data Export**: CSV downloads and image exports

### 4. Camera Calibration System
- **Web-Based Calibration**: Point-and-click zone adjustment with real-time preview
- **Command-Line Tools**: Video, live camera, and image-based calibration
- **Backup/Restore**: Configuration backup and restoration
- **Zone Management**: Individual zone reset and configuration

### 5. Social Media Integration
- **Twitter Bot**: Automated posting of daily summaries and violation alerts
- **Educational Content**: Traffic safety awareness posts
- **Engagement**: Reply to mentions and direct messages
- **Content Generation**: Automated report generation and posting

## 🚀 Quick Start

### Option 1: Web Interface (Recommended)
```bash
# Start the web interface
python3 start_web.py

# Open your browser to http://localhost:5000
# Upload a video and view results
```

### Option 2: Command Line Analysis
```bash
# Install dependencies
pip install -r requirements.txt

# Run analysis on a video file
python3 src/main.py --input path/to/video.mp4 --post-to-twitter

# View results in data/outputs/
```

### Option 3: Docker Deployment
```bash
# Build and run with Docker
docker-compose up -d

# Access at http://localhost:5000
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.12+
- OpenCV dependencies
- FFmpeg (for audio processing)
- Git

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/RedCrowWatch.git
cd RedCrowWatch

# Install Python dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/videos/raw data/outputs logs

# Test the system
python3 test_system.py
```

### Configuration
1. **Camera Setup**: Configure detection zones in `config/config.yaml`
2. **Twitter Integration**: Set up Twitter API credentials (optional)
3. **Calibration**: Use web interface or command-line tools to calibrate camera zones

## 📊 Output & Results

### Analysis Results
- **Video Violations**: Timestamp, type, confidence, location, vehicle type, speed, zone
- **Audio Events**: Timestamp, type, confidence, frequency, duration, decibel level
- **Correlated Events**: Audio-video correlations with strength scores
- **Summary Statistics**: Traffic intensity, safety scores, violation breakdowns

### Visualizations
- **Comprehensive Dashboard**: Multi-panel analysis overview
- **Violation Heatmap**: Geographic distribution of violations
- **Daily Summary**: Summary statistics and trends
- **Interactive Charts**: Doughnut charts, bar charts, line graphs

### Data Export
- **CSV Files**: All analysis data in CSV format
- **Image Files**: Dashboard, heatmap, and summary images
- **API Access**: JSON endpoints for programmatic access

## 🐳 Deployment Options

### Cloud Deployment (Recommended)
- **Railway**: Free tier with automatic GitHub deployment
- **Render**: Free tier with 750 hours/month
- **PythonAnywhere**: Python-focused hosting

### VPS Deployment
- **Docker**: Complete containerization with docker-compose
- **Manual**: Direct Python deployment with systemd

### Local Development
- **Web Interface**: `python3 start_web.py`
- **Docker**: `docker-compose up`
- **Command Line**: Direct Python execution

## 🔧 Technical Stack

### Backend
- **Python 3.12**: Core programming language
- **Flask 3.1.2**: Web framework with blueprint architecture
- **OpenCV 4.12.0.88**: Computer vision and image processing
- **YOLO v8**: State-of-the-art object detection
- **Librosa 0.11.0**: Audio processing and analysis
- **Pandas 2.3.2**: Data manipulation and analysis

### Frontend
- **Bootstrap 5**: Responsive CSS framework
- **Chart.js**: Interactive data visualization
- **JavaScript ES6+**: Modern JavaScript with async/await
- **HTML5**: Semantic markup with accessibility features

### Infrastructure
- **Docker**: Application containerization
- **Gunicorn**: Production WSGI server
- **Railway/Render**: Cloud deployment platforms
- **GitHub**: Source code hosting and CI/CD

## 📁 Project Structure

```
RedCrowWatch/
├── src/                           # Source code
│   ├── analysis/                  # Analysis modules
│   │   ├── integrated_analyzer.py # Multi-modal analysis
│   │   ├── video_analyzer.py      # Computer vision
│   │   ├── audio_analyzer.py      # Audio processing
│   │   ├── nyc_intersection_analyzer.py # NYC-specific analysis
│   │   └── traffic_signal_cycle.py # Traffic signal management
│   ├── visualization/             # Data visualization
│   │   └── traffic_dashboard.py   # Dashboard generation
│   ├── social/                    # Social media integration
│   │   └── twitter_bot.py         # Twitter automation
│   └── calibration/               # Camera calibration
│       ├── camera_calibrator.py   # Command-line calibration
│       └── web_calibrator.py      # Web-based calibration
├── templates/                     # HTML templates
│   ├── base.html                  # Base template
│   ├── index.html                 # Upload page
│   ├── results.html               # Results dashboard
│   └── calibration.html           # Calibration interface
├── static/                        # Static assets
├── data/                          # Data storage
│   ├── videos/raw/               # Uploaded videos
│   ├── videos/processed/         # Processed videos
│   └── outputs/                  # Analysis results
├── config/                        # Configuration
│   └── config.yaml               # Main configuration
├── web_app.py                     # Flask web application
├── start_web.py                   # Web interface startup
├── start.py                       # Railway deployment startup
├── docker-compose.yml             # Docker orchestration
├── Dockerfile                     # Docker container definition
└── requirements.txt               # Python dependencies
```

## 🔒 Security & Compliance

### Security Features
- **File Validation**: Comprehensive file type and size validation
- **Input Sanitization**: Protection against XSS and injection attacks
- **Secure Filenames**: Prevention of path traversal attacks
- **Error Handling**: Graceful error handling without information leakage

### Privacy Considerations
- **Local Processing**: All analysis performed locally
- **Data Retention**: Configurable data retention periods
- **No Personal Data**: System does not collect personal information
- **Public Monitoring**: Designed for public intersection monitoring

## 🚀 Future Enhancements

### Planned Features
- **Real-Time Streaming**: Live video analysis capabilities
- **User Authentication**: Secure user accounts and access control
- **Database Integration**: Persistent data storage with PostgreSQL
- **Mobile App**: Native mobile interface
- **Multi-Camera Support**: Support for multiple camera feeds
- **Advanced Analytics**: Machine learning for pattern recognition

## 📞 Support & Documentation

### Documentation
- **Complete Functionality**: `COMPLETE_FUNCTIONALITY_DOCUMENTATION.md`
- **Tech Stack Breakdown**: `TECH_STACK_BREAKDOWN.md`
- **Camera Calibration**: `CAMERA_CALIBRATION_README.md`
- **Web Interface**: `WEB_INTERFACE_README.md`
- **Deployment Guide**: `DEPLOYMENT_GUIDE.md`

### Troubleshooting
- **System Test**: `python3 test_system.py`
- **Health Check**: `curl http://localhost:5000/health`
- **Logs**: Check `logs/redcrowwatch.log` for detailed information

## 🎯 Use Cases

### Primary Use Cases
- **Traffic Safety Monitoring**: Detect and report traffic violations
- **Intersection Analysis**: Understand traffic patterns and safety concerns
- **Data Collection**: Gather traffic data for safety improvements
- **Public Awareness**: Share traffic safety information via social media

### Secondary Use Cases
- **Research**: Academic research on traffic patterns
- **Urban Planning**: Data for intersection design improvements
- **Community Engagement**: Public awareness and education

## 📈 Performance Metrics

### System Performance
- **Processing Speed**: ~2 frames per second analysis rate
- **File Size Support**: Up to 500MB video files
- **Detection Accuracy**: YOLO v8 with 85%+ accuracy on vehicle detection
- **Audio Detection**: Spectral analysis with configurable thresholds

---

**RedCrowWatch** represents a comprehensive, production-ready traffic monitoring solution that combines cutting-edge AI technology with practical deployment considerations. The system is designed to be both powerful and accessible, making advanced traffic analysis available to communities and organizations working to improve traffic safety.

## Legal Considerations

⚠️ **Important**: This system monitors public intersections. Ensure compliance with:
- Local privacy laws
- Traffic monitoring regulations
- Data retention policies
- Social media platform terms of service

## Contributing

This is a personal project for traffic safety awareness and data collection. Contributions are welcome!

