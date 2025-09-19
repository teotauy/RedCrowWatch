#!/usr/bin/env python3
"""
Setup script for RedCrowWatch Phase 1

This script helps set up the environment and configuration for Phase 1.
"""

import os
import sys
import subprocess
from pathlib import Path
import yaml
import shutil

def main():
    """Main setup function"""
    print("🚦 RedCrowWatch Phase 1 Setup")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Create necessary directories
    print("\n📁 Creating directories...")
    directories = [
        "data/videos/raw",
        "data/videos/processed", 
        "data/outputs",
        "data/debug",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {directory}")
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        sys.exit(1)
    
    # Download YOLO model
    print("\n🤖 Downloading YOLO model...")
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')  # This will download the model
        print("✅ YOLO model downloaded successfully")
    except Exception as e:
        print(f"❌ Failed to download YOLO model: {e}")
        sys.exit(1)
    
    # Setup environment file
    print("\n🔧 Setting up environment...")
    if not Path(".env").exists():
        if Path("env.example").exists():
            shutil.copy("env.example", ".env")
            print("✅ Created .env file from template")
            print("⚠️  Please edit .env file with your API credentials")
        else:
            print("⚠️  No env.example file found")
    else:
        print("✅ .env file already exists")
    
    # Validate configuration
    print("\n⚙️  Validating configuration...")
    try:
        with open("config/config.yaml", 'r') as file:
            config = yaml.safe_load(file)
        print("✅ Configuration file is valid")
    except Exception as e:
        print(f"❌ Configuration file error: {e}")
        sys.exit(1)
    
    # Test imports
    print("\n🧪 Testing imports...")
    try:
        sys.path.append("src")
        from analysis.video_analyzer import VideoAnalyzer
        from visualization.traffic_dashboard import TrafficDashboard
        from social.twitter_bot import TwitterBot
        print("✅ All modules imported successfully")
    except Exception as e:
        print(f"❌ Import error: {e}")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env file with your Twitter API credentials")
    print("2. Update config/config.yaml with your camera settings")
    print("3. Place a test video in data/videos/raw/")
    print("4. Run: python src/main.py --input data/videos/raw/your_video.mp4")
    print("\nFor help, see README.md")

if __name__ == "__main__":
    main()

