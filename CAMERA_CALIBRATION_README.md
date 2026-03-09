# 📷 Camera Calibration Guide

When you remove and reinsert the SD card from your camera, the camera position often shifts slightly, which can throw off the pixel-perfect detection zones. This guide shows you how to quickly recalibrate your camera zones.

## 🚀 Quick Start

### Option 1: Web Interface (Recommended)
1. Start your RedCrowWatch web application
2. Navigate to **Calibrate** in the top menu
3. Use the web interface to adjust zones visually
4. Save your changes

### Option 2: Command Line Tool
```bash
# Run the calibration script
python3 calibrate_camera.py

# Or run directly
python3 src/calibration/camera_calibrator.py --video your_video.mp4
```

## 🎯 When to Calibrate

You should recalibrate your camera when:
- ✅ After removing/reinserting the SD card
- ✅ After repositioning the camera
- ✅ After camera maintenance
- ✅ When detection accuracy decreases
- ✅ After changing camera angle or zoom

## 🛠️ Calibration Methods

### 1. Web Interface (`/calibration/`)
- **Best for**: Quick adjustments, visual feedback
- **Features**: 
  - Visual zone overlay
  - Point-and-click adjustment
  - Real-time preview
  - Individual zone reset
  - Backup/restore functionality

### 2. Command Line Tool
- **Best for**: Batch processing, automation
- **Features**:
  - Video file calibration
  - Live camera calibration
  - Image-based calibration
  - Keyboard shortcuts

### 3. Manual Configuration
- **Best for**: Precise control, advanced users
- **Edit**: `config/config.yaml` directly
- **Backup**: Always backup before changes

## 📋 Step-by-Step Web Calibration

1. **Access the Interface**
   - Go to `http://your-app/calibration/`
   - Or click "Calibrate" in the main navigation

2. **Select a Zone**
   - Use the dropdown to select the zone you want to adjust
   - Each zone represents a different traffic area

3. **Adjust Zone Points**
   - Click and drag zone corner points
   - Add new points by clicking in empty areas
   - Preview changes in real-time

4. **Save Changes**
   - Click "Save Configuration" when satisfied
   - Changes are written to `config/config.yaml`

## 🎮 Keyboard Shortcuts (Command Line)

| Key | Action |
|-----|--------|
| `1-6` | Select specific zone |
| `SPACE` | Next zone |
| `R` | Reset current zone |
| `S` | Save configuration |
| `Q` | Quit without saving |

## 🔧 Zone Types

Your intersection has these detection zones:

1. **Expressway Offramp** - Vehicles merging from highway
2. **One-way Street Approach** - 19th street traffic
3. **One-way Avenue Approach** - 19th avenue traffic  
4. **Intersection Core** - Central intersection area
5. **Bike Lane** - Bicycle detection zone
6. **Pedestrian Crossing** - Pedestrian area

## 📁 File Locations

- **Configuration**: `config/config.yaml`
- **Backup**: `config/config.yaml.backup`
- **Calibration Tool**: `src/calibration/camera_calibrator.py`
- **Web Interface**: `src/calibration/web_calibrator.py`

## 🚨 Troubleshooting

### "Zones not detecting properly"
- Check if camera position changed
- Recalibrate using web interface
- Verify zone coordinates in config

### "Calibration tool won't start"
- Ensure OpenCV is installed: `pip install opencv-python`
- Check video file path is correct
- Verify camera permissions

### "Web interface not loading"
- Check if calibration routes are registered
- Verify template files exist
- Check browser console for errors

## 💡 Tips for Better Calibration

1. **Use a clear video frame** with good lighting
2. **Include all traffic areas** in your zones
3. **Test with actual traffic** after calibration
4. **Keep backups** of working configurations
5. **Calibrate regularly** after camera maintenance

## 🔄 Backup & Restore

### Automatic Backup
- The system automatically creates `.backup` files
- Located in `config/config.yaml.backup`

### Manual Backup
```bash
# Create backup
cp config/config.yaml config/config.yaml.backup.$(date +%Y%m%d)

# Restore from backup
cp config/config.yaml.backup config/config.yaml
```

## 📊 Validation

After calibration, test your setup:

1. **Upload a test video** to the main interface
2. **Check violation detection** accuracy
3. **Verify zone boundaries** are correct
4. **Adjust if needed** using the calibration tools

---

**Need help?** Check the main README or create an issue in the repository.

