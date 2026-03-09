#!/usr/bin/env python3
"""
Quick Camera Calibration Script

Run this script to calibrate your camera zones after repositioning.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

from calibration.camera_calibrator import CameraCalibrator
import logging

def main():
    """Quick calibration interface"""
    print("🚦 RedCrowWatch Camera Calibration Tool")
    print("=" * 50)
    print("This tool helps you adjust detection zones when camera position changes.")
    print("")
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize calibrator
    calibrator = CameraCalibrator()
    
    print("Calibration Options:")
    print("1. Calibrate from video file")
    print("2. Calibrate from live camera")
    print("3. Calibrate from image file")
    print("")
    
    choice = input("Select option (1-3): ").strip()
    
    if choice == '1':
        video_path = input("Enter path to video file: ").strip()
        if os.path.exists(video_path):
            calibrator.calibrate_from_video(video_path)
        else:
            print(f"❌ Video file not found: {video_path}")
    
    elif choice == '2':
        camera_index = input("Enter camera index (default 0): ").strip()
        try:
            camera_index = int(camera_index) if camera_index else 0
            calibrator.calibrate_from_camera(camera_index)
        except ValueError:
            print("❌ Invalid camera index")
    
    elif choice == '3':
        image_path = input("Enter path to image file: ").strip()
        if os.path.exists(image_path):
            calibrator.auto_calibrate_from_image(image_path)
        else:
            print(f"❌ Image file not found: {image_path}")
    
    else:
        print("❌ Invalid choice")
        return
    
    print("")
    print("✅ Calibration complete!")
    print("Your detection zones have been updated in config/config.yaml")

if __name__ == '__main__':
    main()

