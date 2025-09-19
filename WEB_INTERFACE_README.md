# RedCrowWatch Web Interface

A beautiful, modern web interface for uploading and analyzing traffic videos with AI-powered detection.

## 🚀 Quick Start

### 1. Start the Web Interface
```bash
python3 start_web.py
```

### 2. Open Your Browser
Go to: **http://localhost:5000**

### 3. Upload a Video
- Drag and drop your video file, or click to browse
- Supported formats: MP4, AVI, MOV, MKV, WMV
- Maximum file size: 500MB

## ✨ Features

### 🎥 **Video Upload**
- **Drag & Drop Interface** - Simply drag your video file onto the upload area
- **Progress Tracking** - Real-time progress bar with status updates
- **File Validation** - Automatic format and size checking
- **Multiple Formats** - Supports all common video formats

### 📊 **Comprehensive Analysis**
- **Video Analysis** - Detects traffic violations using computer vision
- **Audio Analysis** - Identifies horn honks, sirens, and brake squeals
- **Smart Correlation** - Links audio events with visual violations
- **Safety Metrics** - Calculates traffic intensity and safety scores

### 📈 **Beautiful Results Dashboard**
- **Summary Statistics** - Key metrics at a glance
- **Interactive Charts** - Doughnut charts for violations, bar charts for audio events
- **Data Tables** - Detailed breakdown of all detected events
- **Visualizations** - Downloadable dashboard images and heatmaps
- **Export Options** - Download all data as CSV files

### 🎨 **Modern UI/UX**
- **Responsive Design** - Works on desktop, tablet, and mobile
- **Bootstrap 5** - Modern, clean interface
- **Font Awesome Icons** - Beautiful iconography
- **Chart.js Integration** - Interactive data visualization
- **Real-time Updates** - Live progress tracking and status updates

## 🏗️ Architecture

### **Frontend**
- **HTML5** - Semantic markup with accessibility features
- **Bootstrap 5** - Responsive CSS framework
- **JavaScript** - Interactive functionality and API calls
- **Chart.js** - Data visualization library

### **Backend**
- **Flask** - Lightweight Python web framework
- **File Upload** - Secure file handling with validation
- **API Endpoints** - RESTful API for data exchange
- **Background Processing** - Asynchronous video analysis

### **Integration**
- **RedCrowWatch Core** - Uses existing analysis modules
- **Integrated Analyzer** - Combines video and audio analysis
- **Data Export** - CSV and image file generation

## 📁 File Structure

```
RedCrowWatch/
├── web_app.py              # Main Flask application
├── start_web.py            # Startup script
├── templates/              # HTML templates
│   ├── base.html          # Base template with common elements
│   ├── index.html         # Upload page
│   ├── results.html       # Results dashboard
│   └── error.html         # Error page
├── static/                # Static assets
│   ├── css/               # Custom CSS
│   ├── js/                # JavaScript files
│   └── images/            # Images and icons
└── data/                  # Data storage
    ├── videos/raw/        # Uploaded videos
    └── outputs/           # Analysis results
```

## 🔧 Configuration

### **File Upload Settings**
- Maximum file size: 500MB
- Allowed formats: MP4, AVI, MOV, MKV, WMV
- Upload directory: `data/videos/raw/`

### **Analysis Settings**
- Uses existing `config/config.yaml` configuration
- Supports all NYC intersection detection zones
- Audio analysis enabled by default

### **Web Server Settings**
- Host: 0.0.0.0 (accessible from any IP)
- Port: 5000
- Debug mode: Enabled (for development)

## 📊 Results Dashboard

### **Summary Cards**
- **Video Violations** - Total number of detected violations
- **Audio Events** - Total number of audio events
- **Traffic Intensity** - 0-100 score based on activity
- **Safety Score** - 0-100 score (higher = safer)

### **Detailed Breakdown**
- **Video Violations Table** - Timestamp, type, confidence, vehicle type, zone, speed
- **Audio Events Table** - Timestamp, type, confidence, frequency, duration, decibel level
- **Correlations Table** - Audio-video event correlations with strength scores

### **Visualizations**
- **Comprehensive Dashboard** - Multi-panel analysis overview
- **Violation Heatmap** - Geographic distribution of violations
- **Daily Summary** - Summary statistics and trends

### **Export Options**
- **Individual Files** - Download specific visualizations
- **Bulk Download** - Download all CSV data files
- **API Access** - JSON endpoints for programmatic access

## 🚀 Usage Examples

### **Basic Analysis**
1. Start the web interface: `python3 start_web.py`
2. Open browser to `http://localhost:5000`
3. Upload your Wyze camera video
4. Wait for analysis to complete
5. View results dashboard

### **API Usage**
```bash
# Upload video
curl -X POST -F "file=@video.mp4" http://localhost:5000/upload

# Get results
curl http://localhost:5000/api/results/20240101_120000

# Download file
curl http://localhost:5000/download/20240101_120000/dashboard.png
```

## 🔒 Security Features

- **File Validation** - Checks file type and size
- **Secure Filenames** - Prevents path traversal attacks
- **Input Sanitization** - Protects against XSS attacks
- **Error Handling** - Graceful error handling and user feedback

## 🎯 Performance

- **Asynchronous Processing** - Non-blocking video analysis
- **Progress Tracking** - Real-time status updates
- **Efficient Storage** - Organized file structure
- **Memory Management** - Optimized for large video files

## 🐛 Troubleshooting

### **Common Issues**

1. **"Module not found" errors**
   - Run: `python3 -m pip install flask`

2. **"Permission denied" errors**
   - Check file permissions: `chmod +x start_web.py`

3. **"Port already in use" errors**
   - Kill existing process: `lsof -ti:5000 | xargs kill -9`

4. **"File too large" errors**
   - Reduce video file size or increase limit in `web_app.py`

### **Debug Mode**
- Set `debug=True` in `web_app.py` for detailed error messages
- Check browser console for JavaScript errors
- Monitor server logs for Python errors

## 🚀 Future Enhancements

- **Real-time Streaming** - Live video analysis
- **User Authentication** - Secure user accounts
- **Database Integration** - Persistent data storage
- **Mobile App** - Native mobile interface
- **Cloud Deployment** - Scalable cloud hosting

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review server logs for error details
3. Ensure all dependencies are installed
4. Verify file permissions and directory structure

---

**RedCrowWatch Web Interface** - Making traffic analysis accessible to everyone! 🚦✨
