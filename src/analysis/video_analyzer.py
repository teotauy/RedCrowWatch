"""
Video Analysis Module for Traffic Violation Detection

This module provides the core functionality for analyzing traffic video footage
to detect various types of violations including red light running, speeding,
and wrong-way driving.
"""

import cv2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from ultralytics import YOLO
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Violation:
    """Data class for storing violation information"""
    timestamp: datetime
    violation_type: str
    confidence: float
    location: Tuple[int, int]
    vehicle_id: Optional[str] = None
    speed_mph: Optional[float] = None
    direction: Optional[str] = None


@dataclass
class TrafficLight:
    """Data class for traffic light state"""
    timestamp: datetime
    state: str  # 'red', 'yellow', 'green'
    confidence: float
    position: Tuple[int, int]


class VideoAnalyzer:
    """
    Main class for analyzing traffic video footage and detecting violations
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the video analyzer with configuration"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Load YOLO model for object detection
        self.model = YOLO('yolov8n.pt')  # Nano model for speed
        
        # Initialize tracking variables
        self.tracked_vehicles = {}
        self.traffic_lights = {}
        self.violations = []
        
        # Detection zones from config
        self.detection_zones = self.config['analysis']['detection_zones']
        self.traffic_light_positions = self.config['analysis']['traffic_lights']
        
        # Violation thresholds
        self.violation_config = self.config['analysis']['violations']
        
        logger.info("VideoAnalyzer initialized successfully")
    
    def analyze_video(self, video_path: str) -> List[Violation]:
        """
        Analyze a video file and return detected violations
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of detected violations
        """
        logger.info(f"Starting analysis of video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        logger.info(f"Video info - FPS: {fps}, Frames: {frame_count}, Duration: {duration:.2f}s")
        
        frame_number = 0
        violations = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculate timestamp
                timestamp = datetime.now() + timedelta(seconds=frame_number / fps)
                
                # Process frame
                frame_violations = self._process_frame(frame, timestamp, frame_number)
                violations.extend(frame_violations)
                
                frame_number += 1
                
                # Progress logging
                if frame_number % (fps * 10) == 0:  # Every 10 seconds
                    progress = (frame_number / frame_count) * 100
                    logger.info(f"Progress: {progress:.1f}% - {len(violations)} violations detected")
        
        finally:
            cap.release()
        
        logger.info(f"Analysis complete. Total violations detected: {len(violations)}")
        return violations
    
    def _process_frame(self, frame: np.ndarray, timestamp: datetime, frame_number: int) -> List[Violation]:
        """
        Process a single frame and detect violations
        
        Args:
            frame: Video frame as numpy array
            timestamp: Current timestamp
            frame_number: Frame number in video
            
        Returns:
            List of violations detected in this frame
        """
        violations = []
        
        # Detect objects using YOLO
        results = self.model(frame, verbose=False)
        
        # Extract vehicle detections
        vehicles = self._extract_vehicles(results[0])
        
        # Detect traffic light states
        traffic_light_states = self._detect_traffic_lights(frame)
        
        # Update vehicle tracking
        self._update_vehicle_tracking(vehicles, timestamp)
        
        # Check for violations
        violations.extend(self._check_red_light_violations(vehicles, traffic_light_states, timestamp))
        violations.extend(self._check_speeding_violations(vehicles, timestamp))
        violations.extend(self._check_wrong_way_violations(vehicles, timestamp))
        
        return violations
    
    def _extract_vehicles(self, results) -> List[Dict]:
        """Extract vehicle detections from YOLO results"""
        vehicles = []
        
        for box in results.boxes:
            # Filter for vehicles (car, truck, bus, motorcycle)
            class_id = int(box.cls[0])
            class_name = results.names[class_id]
            
            if class_name in ['car', 'truck', 'bus', 'motorcycle']:
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                vehicles.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                })
        
        return vehicles
    
    def _detect_traffic_lights(self, frame: np.ndarray) -> Dict[str, str]:
        """
        Detect traffic light states using color detection
        
        This is a simplified implementation. In production, you might want to use
        a more sophisticated approach or train a custom model for traffic lights.
        """
        traffic_light_states = {}
        
        for light_config in self.traffic_light_positions:
            name = light_config['name']
            position = light_config['position']
            radius = light_config['detection_radius']
            
            # Extract region around traffic light
            x, y = position
            roi = frame[max(0, y-radius):min(frame.shape[0], y+radius),
                       max(0, x-radius):min(frame.shape[1], x+radius)]
            
            if roi.size > 0:
                # Convert to HSV for better color detection
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                
                # Define color ranges for traffic lights
                red_lower = np.array([0, 50, 50])
                red_upper = np.array([10, 255, 255])
                yellow_lower = np.array([20, 50, 50])
                yellow_upper = np.array([30, 255, 255])
                green_lower = np.array([40, 50, 50])
                green_upper = np.array([80, 255, 255])
                
                # Check for each color
                red_mask = cv2.inRange(hsv, red_lower, red_upper)
                yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
                green_mask = cv2.inRange(hsv, green_lower, green_upper)
                
                # Determine which color is most prominent
                red_pixels = cv2.countNonZero(red_mask)
                yellow_pixels = cv2.countNonZero(yellow_mask)
                green_pixels = cv2.countNonZero(green_mask)
                
                if red_pixels > yellow_pixels and red_pixels > green_pixels:
                    traffic_light_states[name] = 'red'
                elif yellow_pixels > red_pixels and yellow_pixels > green_pixels:
                    traffic_light_states[name] = 'yellow'
                elif green_pixels > red_pixels and green_pixels > yellow_pixels:
                    traffic_light_states[name] = 'green'
                else:
                    traffic_light_states[name] = 'unknown'
        
        return traffic_light_states
    
    def _update_vehicle_tracking(self, vehicles: List[Dict], timestamp: datetime):
        """Update vehicle tracking for speed and direction analysis"""
        # This is a simplified tracking implementation
        # In production, you'd want to use a proper tracking algorithm like DeepSORT
        
        for vehicle in vehicles:
            center = vehicle['center']
            vehicle_id = f"vehicle_{center[0]}_{center[1]}"  # Simple ID based on position
            
            if vehicle_id in self.tracked_vehicles:
                # Update existing vehicle
                prev_center = self.tracked_vehicles[vehicle_id]['center']
                prev_time = self.tracked_vehicles[vehicle_id]['timestamp']
                
                # Calculate movement
                distance = np.sqrt((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2)
                time_diff = (timestamp - prev_time).total_seconds()
                
                if time_diff > 0:
                    speed_pixels_per_second = distance / time_diff
                    # Convert to mph (this would need calibration based on camera setup)
                    speed_mph = speed_pixels_per_second * 0.1  # Rough conversion
                    
                    self.tracked_vehicles[vehicle_id].update({
                        'center': center,
                        'timestamp': timestamp,
                        'speed_mph': speed_mph,
                        'direction': self._calculate_direction(prev_center, center)
                    })
            else:
                # New vehicle
                self.tracked_vehicles[vehicle_id] = {
                    'center': center,
                    'timestamp': timestamp,
                    'speed_mph': 0,
                    'direction': 'unknown',
                    'class': vehicle['class']
                }
    
    def _calculate_direction(self, prev_center: List[int], current_center: List[int]) -> str:
        """Calculate vehicle direction based on movement"""
        dx = current_center[0] - prev_center[0]
        dy = current_center[1] - prev_center[1]
        
        if abs(dx) > abs(dy):
            return 'eastbound' if dx > 0 else 'westbound'
        else:
            return 'southbound' if dy > 0 else 'northbound'
    
    def _check_red_light_violations(self, vehicles: List[Dict], traffic_light_states: Dict[str, str], 
                                  timestamp: datetime) -> List[Violation]:
        """Check for red light running violations"""
        violations = []
        
        # Check if any traffic light is red
        red_lights = [name for name, state in traffic_light_states.items() if state == 'red']
        
        if red_lights:
            for vehicle in vehicles:
                center = vehicle['center']
                
                # Check if vehicle is in intersection zones during red light
                for zone in self.detection_zones:
                    if self._point_in_polygon(center, zone['coordinates']):
                        violation = Violation(
                            timestamp=timestamp,
                            violation_type='red_light_running',
                            confidence=vehicle['confidence'],
                            location=tuple(center),
                            vehicle_id=f"vehicle_{center[0]}_{center[1]}"
                        )
                        violations.append(violation)
                        break
        
        return violations
    
    def _check_speeding_violations(self, vehicles: List[Dict], timestamp: datetime) -> List[Violation]:
        """Check for speeding violations"""
        violations = []
        speed_limit = self.violation_config['speeding']['speed_limit_mph']
        
        for vehicle in vehicles:
            center = vehicle['center']
            vehicle_id = f"vehicle_{center[0]}_{center[1]}"
            
            if vehicle_id in self.tracked_vehicles:
                speed = self.tracked_vehicles[vehicle_id]['speed_mph']
                
                if speed > speed_limit:
                    violation = Violation(
                        timestamp=timestamp,
                        violation_type='speeding',
                        confidence=vehicle['confidence'],
                        location=tuple(center),
                        vehicle_id=vehicle_id,
                        speed_mph=speed
                    )
                    violations.append(violation)
        
        return violations
    
    def _check_wrong_way_violations(self, vehicles: List[Dict], timestamp: datetime) -> List[Violation]:
        """Check for wrong-way driving violations"""
        violations = []
        
        for vehicle in vehicles:
            center = vehicle['center']
            vehicle_id = f"vehicle_{center[0]}_{center[1]}"
            
            if vehicle_id in self.tracked_vehicles:
                direction = self.tracked_vehicles[vehicle_id]['direction']
                
                # Check if vehicle is going wrong way in specific zones
                # This would need to be customized based on your intersection layout
                for zone in self.detection_zones:
                    if self._point_in_polygon(center, zone['coordinates']):
                        expected_direction = zone['name'].split('_')[0]  # Extract direction from zone name
                        
                        if direction != expected_direction and direction != 'unknown':
                            violation = Violation(
                                timestamp=timestamp,
                                violation_type='wrong_way',
                                confidence=vehicle['confidence'],
                                location=tuple(center),
                                vehicle_id=vehicle_id,
                                direction=direction
                            )
                            violations.append(violation)
                        break
        
        return violations
    
    def _point_in_polygon(self, point: List[int], polygon: List[List[int]]) -> bool:
        """Check if a point is inside a polygon using ray casting algorithm"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def save_violations_to_csv(self, violations: List[Violation], output_path: str):
        """Save violations to CSV file"""
        data = []
        for violation in violations:
            data.append({
                'timestamp': violation.timestamp,
                'violation_type': violation.violation_type,
                'confidence': violation.confidence,
                'location_x': violation.location[0],
                'location_y': violation.location[1],
                'vehicle_id': violation.vehicle_id,
                'speed_mph': violation.speed_mph,
                'direction': violation.direction
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(violations)} violations to {output_path}")


if __name__ == "__main__":
    # Example usage
    analyzer = VideoAnalyzer()
    
    # Analyze a video file
    video_path = "data/videos/raw/sample_video.mp4"
    violations = analyzer.analyze_video(video_path)
    
    # Save results
    output_path = "data/outputs/violations.csv"
    analyzer.save_violations_to_csv(violations, output_path)
    
    print(f"Analysis complete. Found {len(violations)} violations.")

