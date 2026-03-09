#!/usr/bin/env python3
"""
Camera Calibration Tool for RedCrowWatch

This tool helps adjust detection zones when camera position changes
after SD card removal/reinsertion or camera repositioning.
"""

import cv2
import numpy as np
import yaml
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


class CameraCalibrator:
    """
    Interactive camera calibration tool for adjusting detection zones
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the camera calibrator"""
        self.config_path = config_path
        self.load_config()
        self.current_zone_index = 0
        self.dragging = False
        self.drag_point_index = -1
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
        
        logger.info(f"Camera calibrator initialized with {len(self.zones)} zones")
    
    def load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            sys.exit(1)
    
    def save_config(self, backup: bool = True):
        """Save updated configuration"""
        if backup:
            # Create backup of original config
            backup_path = f"{self.config_path}.backup"
            with open(backup_path, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False)
            logger.info(f"Backup saved to {backup_path}")
        
        # Update zones in config
        self.config['analysis']['detection_zones'] = self.zones
        
        # Save updated config
        with open(self.config_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False)
        
        logger.info(f"Updated configuration saved to {self.config_path}")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for zone editing"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicking on a zone point
            for zone_idx, zone in enumerate(self.zones):
                for point_idx, point in enumerate(zone['coordinates']):
                    if abs(point[0] - x) < 10 and abs(point[1] - y) < 10:
                        self.current_zone_index = zone_idx
                        self.drag_point_index = point_idx
                        self.dragging = True
                        return
            
            # If not clicking on existing point, add new point to current zone
            if self.current_zone_index < len(self.zones):
                self.zones[self.current_zone_index]['coordinates'].append([x, y])
                self.drag_point_index = len(self.zones[self.current_zone_index]['coordinates']) - 1
                self.dragging = True
        
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            # Update dragged point
            if (self.current_zone_index < len(self.zones) and 
                self.drag_point_index < len(self.zones[self.current_zone_index]['coordinates'])):
                self.zones[self.current_zone_index]['coordinates'][self.drag_point_index] = [x, y]
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.drag_point_index = -1
    
    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
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
        
        # Highlight current zone
        if self.current_zone_index < len(self.zones):
            current_zone = self.zones[self.current_zone_index]
            coordinates = np.array(current_zone['coordinates'], np.int32)
            if len(coordinates) >= 3:
                cv2.polylines(overlay, [coordinates], True, (0, 255, 255), 4)
        
        # Blend overlay with original frame
        alpha = 0.3
        return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
    
    def draw_instructions(self, frame: np.ndarray) -> np.ndarray:
        """Draw calibration instructions on frame"""
        instructions = [
            "CAMERA CALIBRATION MODE",
            "",
            f"Current Zone: {self.zones[self.current_zone_index]['name'] if self.current_zone_index < len(self.zones) else 'None'}",
            "",
            "Controls:",
            "1-6: Select zone",
            "SPACE: Next zone",
            "R: Reset current zone",
            "S: Save configuration",
            "Q: Quit without saving",
            "",
            "Mouse:",
            "Click: Add/move zone points",
            "Drag: Move existing points"
        ]
        
        y_offset = 30
        for instruction in instructions:
            cv2.putText(frame, instruction, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
        
        return frame
    
    def calibrate_from_video(self, video_path: str):
        """Run interactive calibration from video file"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return
        
        # Get first frame
        ret, frame = cap.read()
        if not ret:
            logger.error("Could not read video frame")
            return
        
        # Create window and set mouse callback
        cv2.namedWindow('Camera Calibration', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Camera Calibration', self.mouse_callback)
        
        logger.info("Starting camera calibration...")
        logger.info("Use mouse to adjust zone points, keyboard for controls")
        
        while True:
            # Draw zones and instructions
            display_frame = self.draw_zones(frame.copy())
            display_frame = self.draw_instructions(display_frame)
            
            cv2.imshow('Camera Calibration', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                logger.info("Calibration cancelled")
                break
            elif key == ord('s'):
                self.save_config()
                logger.info("Configuration saved!")
                break
            elif key == ord('r'):
                # Reset current zone
                if self.current_zone_index < len(self.zones):
                    self.zones[self.current_zone_index] = self.original_zones[self.current_zone_index].copy()
                    logger.info(f"Reset zone: {self.zones[self.current_zone_index]['name']}")
            elif key == ord(' '):
                # Next zone
                self.current_zone_index = (self.current_zone_index + 1) % len(self.zones)
                logger.info(f"Selected zone: {self.zones[self.current_zone_index]['name']}")
            elif key >= ord('1') and key <= ord('6'):
                # Select specific zone
                zone_num = key - ord('1')
                if zone_num < len(self.zones):
                    self.current_zone_index = zone_num
                    logger.info(f"Selected zone: {self.zones[self.current_zone_index]['name']}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def calibrate_from_camera(self, camera_index: int = 0):
        """Run interactive calibration from live camera"""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            logger.error(f"Could not open camera: {camera_index}")
            return
        
        # Create window and set mouse callback
        cv2.namedWindow('Camera Calibration', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Camera Calibration', self.mouse_callback)
        
        logger.info("Starting live camera calibration...")
        logger.info("Use mouse to adjust zone points, keyboard for controls")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Could not read camera frame")
                break
            
            # Draw zones and instructions
            display_frame = self.draw_zones(frame.copy())
            display_frame = self.draw_instructions(display_frame)
            
            cv2.imshow('Camera Calibration', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                logger.info("Calibration cancelled")
                break
            elif key == ord('s'):
                self.save_config()
                logger.info("Configuration saved!")
                break
            elif key == ord('r'):
                # Reset current zone
                if self.current_zone_index < len(self.zones):
                    self.zones[self.current_zone_index] = self.original_zones[self.current_zone_index].copy()
                    logger.info(f"Reset zone: {self.zones[self.current_zone_index]['name']}")
            elif key == ord(' '):
                # Next zone
                self.current_zone_index = (self.current_zone_index + 1) % len(self.zones)
                logger.info(f"Selected zone: {self.zones[self.current_zone_index]['name']}")
            elif key >= ord('1') and key <= ord('6'):
                # Select specific zone
                zone_num = key - ord('1')
                if zone_num < len(self.zones):
                    self.current_zone_index = zone_num
                    logger.info(f"Selected zone: {self.zones[self.current_zone_index]['name']}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def auto_calibrate_from_image(self, image_path: str):
        """Semi-automatic calibration using image analysis"""
        frame = cv2.imread(image_path)
        if frame is None:
            logger.error(f"Could not load image: {image_path}")
            return
        
        # Create window and set mouse callback
        cv2.namedWindow('Auto Calibration', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Auto Calibration', self.mouse_callback)
        
        logger.info("Starting auto calibration...")
        logger.info("Click on key points to define zones")
        
        while True:
            display_frame = self.draw_zones(frame.copy())
            display_frame = self.draw_instructions(display_frame)
            
            cv2.imshow('Auto Calibration', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                logger.info("Calibration cancelled")
                break
            elif key == ord('s'):
                self.save_config()
                logger.info("Configuration saved!")
                break
            elif key == ord('r'):
                # Reset current zone
                if self.current_zone_index < len(self.zones):
                    self.zones[self.current_zone_index] = self.original_zones[self.current_zone_index].copy()
                    logger.info(f"Reset zone: {self.zones[self.current_zone_index]['name']}")
            elif key == ord(' '):
                # Next zone
                self.current_zone_index = (self.current_zone_index + 1) % len(self.zones)
                logger.info(f"Selected zone: {self.zones[self.current_zone_index]['name']}")
        
        cv2.destroyAllWindows()


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Camera Calibration Tool for RedCrowWatch')
    parser.add_argument('--config', '-c', default='config/config.yaml', 
                       help='Configuration file path')
    parser.add_argument('--video', '-v', help='Video file for calibration')
    parser.add_argument('--camera', '-cam', type=int, default=0, 
                       help='Camera index for live calibration')
    parser.add_argument('--image', '-i', help='Image file for calibration')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    calibrator = CameraCalibrator(args.config)
    
    if args.video:
        calibrator.calibrate_from_video(args.video)
    elif args.image:
        calibrator.auto_calibrate_from_image(args.image)
    else:
        calibrator.calibrate_from_camera(args.camera)


if __name__ == '__main__':
    main()

