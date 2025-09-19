#!/usr/bin/env python3
"""
Test script for RedCrowWatch NYC Intersection System
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

def test_imports():
    """Test all imports"""
    print("🧪 Testing imports...")
    
    try:
        from analysis.integrated_analyzer import IntegratedAnalyzer
        print("✅ Integrated Analyzer imported")
    except Exception as e:
        print(f"❌ Integrated Analyzer import failed: {e}")
        return False
    
    try:
        from visualization.traffic_dashboard import TrafficDashboard
        print("✅ Traffic Dashboard imported")
    except Exception as e:
        print(f"❌ Traffic Dashboard import failed: {e}")
        return False
    
    try:
        from social.twitter_bot import TwitterBot
        print("✅ Twitter Bot imported")
    except Exception as e:
        print(f"❌ Twitter Bot import failed: {e}")
        return False
    
    return True

def test_config():
    """Test configuration loading"""
    print("\n⚙️  Testing configuration...")
    
    try:
        import yaml
        with open('config/config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        
        print("✅ Configuration loaded successfully")
        print(f"  - Detection zones: {len(config['analysis']['detection_zones'])}")
        print(f"  - Traffic lights: {len(config['analysis']['traffic_lights'])}")
        print(f"  - Violation types: {len(config['analysis']['violations'])}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_analyzer_initialization():
    """Test analyzer initialization"""
    print("\n🤖 Testing integrated analyzer initialization...")
    
    try:
        from analysis.integrated_analyzer import IntegratedAnalyzer
        analyzer = IntegratedAnalyzer()
        print("✅ Integrated Analyzer initialized")
        print(f"  - Video detection zones: {len(analyzer.video_analyzer.detection_zones)}")
        print(f"  - Traffic lights: {len(analyzer.video_analyzer.traffic_lights)}")
        print(f"  - Audio analysis enabled: {analyzer.audio_analyzer.audio_config.get('enabled', False)}")
        return True
    except Exception as e:
        print(f"❌ Analyzer initialization failed: {e}")
        return False

def test_dashboard_initialization():
    """Test dashboard initialization"""
    print("\n📊 Testing dashboard initialization...")
    
    try:
        from visualization.traffic_dashboard import TrafficDashboard
        dashboard = TrafficDashboard()
        print("✅ Traffic Dashboard initialized")
        return True
    except Exception as e:
        print(f"❌ Dashboard initialization failed: {e}")
        return False

def test_yolo_model():
    """Test YOLO model loading"""
    print("\n🎯 Testing YOLO model...")
    
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        print("✅ YOLO model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ YOLO model test failed: {e}")
        return False

def test_audio_libraries():
    """Test audio processing libraries"""
    print("\n🔊 Testing audio libraries...")
    
    try:
        import librosa
        import scipy
        import soundfile
        print("✅ Audio libraries imported successfully")
        return True
    except Exception as e:
        print(f"❌ Audio libraries test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚦 RedCrowWatch NYC Intersection System Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config,
        test_analyzer_initialization,
        test_dashboard_initialization,
        test_yolo_model,
        test_audio_libraries
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Place a video file in data/videos/raw/")
        print("2. Run: python3 src/main.py --input data/videos/raw/your_video.mp4")
        print("3. Check results in data/outputs/")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
