#!/usr/bin/env python3
"""
Web-based Camera Calibration Interface

Provides a web interface for camera calibration that can be accessed
through the main web application.
"""

import cv2
import numpy as np
import yaml
import base64
import io
from flask import Blueprint, render_template, request, jsonify, send_file
from typing import Dict, List, Tuple
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Create blueprint for calibration routes
calibration_bp = Blueprint('calibration', __name__, url_prefix='/calibration')

class WebCalibrator:
    """Web-based camera calibration interface"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.load_config()
        self.zones = self.config['analysis']['detection_zones'].copy()
        self.original_zones = self.zones.copy()
        
        # Colors for different zones
        self.zone_colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
    
    def load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            self.config = {}
    
    def save_config(self):
        """Save updated configuration"""
        # Update zones in config
        self.config['analysis']['detection_zones'] = self.zones
        
        # Save updated config
        with open(self.config_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False)
        
        logger.info(f"Updated configuration saved to {self.config_path}")
    
    def draw_zones_on_frame(self, frame: np.ndarray) -> np.ndarray:
        """Draw detection zones on frame"""
        overlay = frame.copy()
        
        for zone_idx, zone in enumerate(self.zones):
            color = self.zone_colors[zone_idx % len(self.zone_colors)]
            coordinates = np.array(zone['coordinates'], np.int32)
            
            # Draw zone polygon
            if len(coordinates) >= 3:
                cv2.fillPoly(overlay, [coordinates], color)
                cv2.polylines(overlay, [coordinates], True, (255, 255, 255), 2)
            
            # Draw zone points
            for point in zone['coordinates']:
                cv2.circle(overlay, tuple(point), 5, (255, 255, 255), -1)
                cv2.circle(overlay, tuple(point), 3, color, -1)
            
            # Draw zone label
            if coordinates.size > 0:
                center = np.mean(coordinates, axis=0).astype(int)
                cv2.putText(overlay, zone['name'], tuple(center), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Blend overlay with original frame
        alpha = 0.3
        return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
    
    def frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert OpenCV frame to base64 string for web display"""
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{frame_base64}"

# Global calibrator instance
web_calibrator = WebCalibrator()

@calibration_bp.route('/')
def calibration_page():
    """Serve calibration page"""
    return render_template('calibration.html', zones=web_calibrator.zones)

@calibration_bp.route('/get_zones')
def get_zones():
    """Get current zone configuration"""
    return jsonify({
        'zones': web_calibrator.zones,
        'colors': web_calibrator.zone_colors
    })

@calibration_bp.route('/update_zone', methods=['POST'])
def update_zone():
    """Update a specific zone"""
    try:
        data = request.json
        zone_index = data.get('zone_index')
        coordinates = data.get('coordinates')
        
        if zone_index is not None and coordinates is not None:
            if 0 <= zone_index < len(web_calibrator.zones):
                web_calibrator.zones[zone_index]['coordinates'] = coordinates
                return jsonify({'success': True, 'message': 'Zone updated'})
        
        return jsonify({'success': False, 'message': 'Invalid zone data'}), 400
    
    except Exception as e:
        logger.error(f"Error updating zone: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@calibration_bp.route('/reset_zone', methods=['POST'])
def reset_zone():
    """Reset a specific zone to original coordinates"""
    try:
        data = request.json
        zone_index = data.get('zone_index')
        
        if zone_index is not None and 0 <= zone_index < len(web_calibrator.zones):
            web_calibrator.zones[zone_index] = web_calibrator.original_zones[zone_index].copy()
            return jsonify({'success': True, 'message': 'Zone reset'})
        
        return jsonify({'success': False, 'message': 'Invalid zone index'}), 400
    
    except Exception as e:
        logger.error(f"Error resetting zone: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@calibration_bp.route('/save_config', methods=['POST'])
def save_config():
    """Save current configuration"""
    try:
        web_calibrator.save_config()
        return jsonify({'success': True, 'message': 'Configuration saved'})
    
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@calibration_bp.route('/backup_config', methods=['POST'])
def backup_config():
    """Create a backup of current configuration"""
    try:
        import shutil
        from datetime import datetime
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = f"{web_calibrator.config_path}.backup.{timestamp}"
        
        shutil.copy2(web_calibrator.config_path, backup_path)
        
        return jsonify({
            'success': True, 
            'message': f'Backup created: {backup_path}',
            'backup_path': backup_path
        })
    
    except Exception as e:
        logger.error(f"Error creating backup: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@calibration_bp.route('/restore_config', methods=['POST'])
def restore_config():
    """Restore configuration from backup"""
    try:
        data = request.json
        backup_path = data.get('backup_path')
        
        if not backup_path or not os.path.exists(backup_path):
            return jsonify({'success': False, 'message': 'Invalid backup path'}), 400
        
        # Restore from backup
        shutil.copy2(backup_path, web_calibrator.config_path)
        
        # Reload configuration
        web_calibrator.load_config()
        web_calibrator.zones = web_calibrator.config['analysis']['detection_zones'].copy()
        web_calibrator.original_zones = web_calibrator.zones.copy()
        
        return jsonify({
            'success': True, 
            'message': 'Configuration restored',
            'zones': web_calibrator.zones
        })
    
    except Exception as e:
        logger.error(f"Error restoring config: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@calibration_bp.route('/list_backups')
def list_backups():
    """List available configuration backups"""
    try:
        config_dir = os.path.dirname(web_calibrator.config_path)
        config_name = os.path.basename(web_calibrator.config_path)
        
        backups = []
        for file in os.listdir(config_dir):
            if file.startswith(f"{config_name}.backup"):
                file_path = os.path.join(config_dir, file)
                stat = os.stat(file_path)
                backups.append({
                    'filename': file,
                    'path': file_path,
                    'size': stat.st_size,
                    'modified': stat.st_mtime
                })
        
        # Sort by modification time (newest first)
        backups.sort(key=lambda x: x['modified'], reverse=True)
        
        return jsonify({'success': True, 'backups': backups})
    
    except Exception as e:
        logger.error(f"Error listing backups: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@calibration_bp.route('/preview_frame', methods=['POST'])
def preview_frame():
    """Generate preview frame with zones"""
    try:
        # This would typically get a frame from camera or video
        # For now, create a sample frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (50, 50, 50)  # Dark gray background
        
        # Add some sample content
        cv2.putText(frame, "Camera Calibration Preview", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Adjust zones using the interface below", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Draw zones
        frame_with_zones = web_calibrator.draw_zones_on_frame(frame)
        
        # Convert to base64
        frame_base64 = web_calibrator.frame_to_base64(frame_with_zones)
        
        return jsonify({
            'success': True,
            'frame': frame_base64
        })
    
    except Exception as e:
        logger.error(f"Error generating preview: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

def register_calibration_routes(app):
    """Register calibration routes with Flask app"""
    app.register_blueprint(calibration_bp)
    logger.info("Calibration routes registered")
