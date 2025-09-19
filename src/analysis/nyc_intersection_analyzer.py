"""
NYC Intersection Analyzer - Specialized for T-intersection with expressway offramp

This module provides specialized analysis for the unique intersection layout:
- T-intersection with one-way streets
- Expressway offramp merging
- Bike lanes and pedestrian bridge terminus
- Pedestrian signal as traffic light proxy
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
class NYCViolation:
    """Enhanced violation class for NYC intersection"""
    timestamp: datetime
    violation_type: str
    confidence: float
    location: Tuple[int, int]
    vehicle_id: Optional[str] = None
    speed_mph: Optional[float] = None
    direction: Optional[str] = None
    vehicle_type: Optional[str] = None  # car, truck, bus, bike
    zone: Optional[str] = None  # which detection zone


class NYCIntersectionAnalyzer:
    """
    Specialized analyzer for NYC T-intersection with expressway offramp
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the NYC intersection analyzer"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Load YOLO model
        self.model = YOLO('yolov8n.pt')
        
        # Initialize tracking
        self.tracked_vehicles = {}
        self.pedestrian_signal_state = "unknown"
        self.violations = []
        
        # NYC-specific settings
        self.detection_zones = self.config['analysis']['detection_zones']
        self.traffic_lights = self.config['analysis']['traffic_lights']
        self.violation_config = self.config['analysis']['violations']
        
        # Vehicle type mapping for NYC
        self.vehicle_types = {
            'car': 'car',
            'truck': 'truck', 
            'bus': 'bus',
            'motorcycle': 'motorcycle',
            'bicycle': 'bicycle'
        }
        
        logger.info("NYC Intersection Analyzer initialized")
    
    def analyze_video(self, video_path: str) -> List[NYCViolation]:
        """Analyze video for NYC intersection violations"""
        logger.info(f"Starting NYC intersection analysis: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video: {fps} FPS, {frame_count} frames")
        
        frame_number = 0
        violations = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = datetime.now() + timedelta(seconds=frame_number / fps)
                
                # Process frame for NYC intersection
                frame_violations = self._process_nyc_frame(frame, timestamp, frame_number)
                violations.extend(frame_violations)
                
                frame_number += 1
                
                # Progress logging
                if frame_number % (fps * 10) == 0:
                    progress = (frame_number / frame_count) * 100
                    logger.info(f"Progress: {progress:.1f}% - {len(violations)} violations")
        
        finally:
            cap.release()
        
        logger.info(f"NYC analysis complete: {len(violations)} violations detected")
        return violations
    
    def _process_nyc_frame(self, frame: np.ndarray, timestamp: datetime, frame_number: int) -> List[NYCViolation]:
        """Process frame for NYC intersection-specific violations"""
        violations = []
        
        # Detect objects
        results = self.model(frame, verbose=False)
        vehicles = self._extract_vehicles(results[0])
        
        # Detect pedestrian signal state
        self._detect_pedestrian_signal(frame)
        
        # Update vehicle tracking
        self._update_vehicle_tracking(vehicles, timestamp)
        
        # Check for NYC-specific violations
        violations.extend(self._check_red_light_violations_nyc(vehicles, timestamp))
        violations.extend(self._check_speeding_violations_nyc(vehicles, timestamp))
        violations.extend(self._check_bike_lane_violations(vehicles, timestamp))
        violations.extend(self._check_pedestrian_violations(vehicles, timestamp))
        violations.extend(self._check_bus_lane_violations(vehicles, timestamp))
        violations.extend(self._check_expressway_offramp_violations(vehicles, timestamp))
        
        return violations
    
    def _extract_vehicles(self, results) -> List[Dict]:
        """Extract vehicles with NYC-specific classification"""
        vehicles = []
        
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = results.names[class_id]
            
            if class_name in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']:
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Determine vehicle type for NYC context
                vehicle_type = self._classify_nyc_vehicle(class_name, confidence, x2-x1, y2-y1)
                
                vehicles.append({
                    'class': class_name,
                    'vehicle_type': vehicle_type,
                    'confidence': confidence,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                    'size': [int(x2-x1), int(y2-y1)]
                })
        
        return vehicles
    
    def _classify_nyc_vehicle(self, class_name: str, confidence: float, width: float, height: float) -> str:
        """Classify vehicle type for NYC context"""
        # Size-based classification for NYC
        area = width * height
        
        if class_name == 'bus':
            return 'bus'
        elif class_name == 'truck' or (class_name == 'car' and area > 50000):
            return 'truck'
        elif class_name == 'bicycle':
            return 'bicycle'
        elif class_name == 'motorcycle':
            return 'motorcycle'
        else:
            return 'car'
    
    def _detect_pedestrian_signal(self, frame: np.ndarray):
        """Detect pedestrian signal state (walk/don't walk)"""
        # Find pedestrian signal from config
        pedestrian_signal = None
        for light in self.traffic_lights:
            if light['type'] == 'pedestrian':
                pedestrian_signal = light
                break
        
        if not pedestrian_signal:
            return
        
        # Extract region around pedestrian signal
        x, y = pedestrian_signal['position']
        radius = pedestrian_signal['detection_radius']
        
        roi = frame[max(0, y-radius):min(frame.shape[0], y+radius),
                   max(0, x-radius):min(frame.shape[1], x+radius)]
        
        if roi.size > 0:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Look for walk signal (white/light) vs don't walk (red/dark)
            # This is a simplified approach - in production you'd want more sophisticated detection
            
            # Check for bright/white areas (walk signal)
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
            white_pixels = cv2.countNonZero(white_mask)
            
            # Check for red areas (don't walk signal)
            lower_red = np.array([0, 50, 50])
            upper_red = np.array([10, 255, 255])
            red_mask = cv2.inRange(hsv, lower_red, upper_red)
            red_pixels = cv2.countNonZero(red_mask)
            
            # Determine state
            if white_pixels > red_pixels and white_pixels > 50:
                self.pedestrian_signal_state = "walk"
            elif red_pixels > white_pixels and red_pixels > 50:
                self.pedestrian_signal_state = "don't_walk"
            else:
                self.pedestrian_signal_state = "unknown"
    
    def _update_vehicle_tracking(self, vehicles: List[Dict], timestamp: datetime):
        """Update vehicle tracking with NYC-specific logic"""
        for vehicle in vehicles:
            center = vehicle['center']
            vehicle_id = f"{vehicle['vehicle_type']}_{center[0]}_{center[1]}"
            
            if vehicle_id in self.tracked_vehicles:
                # Update existing vehicle
                prev_data = self.tracked_vehicles[vehicle_id]
                prev_center = prev_data['center']
                prev_time = prev_data['timestamp']
                
                # Calculate movement
                distance = np.sqrt((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2)
                time_diff = (timestamp - prev_time).total_seconds()
                
                if time_diff > 0:
                    speed_pixels_per_second = distance / time_diff
                    # Convert to mph (calibrated for NYC intersection)
                    speed_mph = speed_pixels_per_second * 0.08  # Calibration factor
                    
                    # Determine direction based on movement
                    direction = self._determine_nyc_direction(prev_center, center)
                    
                    # Determine which zone the vehicle is in
                    zone = self._get_vehicle_zone(center)
                    
                    self.tracked_vehicles[vehicle_id].update({
                        'center': center,
                        'timestamp': timestamp,
                        'speed_mph': speed_mph,
                        'direction': direction,
                        'zone': zone,
                        'vehicle_type': vehicle['vehicle_type']
                    })
            else:
                # New vehicle
                zone = self._get_vehicle_zone(center)
                self.tracked_vehicles[vehicle_id] = {
                    'center': center,
                    'timestamp': timestamp,
                    'speed_mph': 0,
                    'direction': 'unknown',
                    'zone': zone,
                    'vehicle_type': vehicle['vehicle_type'],
                    'class': vehicle['class']
                }
    
    def _determine_nyc_direction(self, prev_center: List[int], current_center: List[int]) -> str:
        """Determine direction based on NYC intersection layout"""
        dx = current_center[0] - prev_center[0]
        dy = current_center[1] - prev_center[1]
        
        # Based on T-intersection layout
        if abs(dx) > abs(dy):
            if dx > 0:
                return 'eastbound'  # Towards expressway offramp
            else:
                return 'westbound'  # Away from expressway
        else:
            if dy > 0:
                return 'southbound'  # Down the one-way avenue
            else:
                return 'northbound'  # Up the one-way avenue
    
    def _get_vehicle_zone(self, center: List[int]) -> str:
        """Determine which detection zone the vehicle is in"""
        for zone in self.detection_zones:
            if self._point_in_polygon(center, zone['coordinates']):
                return zone['name']
        return 'unknown'
    
    def _check_red_light_violations_nyc(self, vehicles: List[Dict], timestamp: datetime) -> List[NYCViolation]:
        """Check for red light violations using pedestrian signal as proxy"""
        violations = []
        
        # Use pedestrian signal as proxy for traffic light state
        if self.pedestrian_signal_state == "don't_walk":
            # Assume red light when pedestrian signal shows don't walk
            for vehicle in vehicles:
                center = vehicle['center']
                zone = self._get_vehicle_zone(center)
                
                # Check if vehicle is in intersection core during "red light"
                if zone == 'intersection_core':
                    violation = NYCViolation(
                        timestamp=timestamp,
                        violation_type='red_light_running',
                        confidence=vehicle['confidence'],
                        location=tuple(center),
                        vehicle_id=f"{vehicle['vehicle_type']}_{center[0]}_{center[1]}",
                        vehicle_type=vehicle['vehicle_type'],
                        zone=zone
                    )
                    violations.append(violation)
        
        return violations
    
    def _check_speeding_violations_nyc(self, vehicles: List[Dict], timestamp: datetime) -> List[NYCViolation]:
        """Check for speeding violations with NYC-specific thresholds"""
        violations = []
        speed_limit = self.violation_config['speeding']['speed_limit_mph']
        truck_penalty = self.violation_config['speeding']['truck_route_penalty']
        
        for vehicle in vehicles:
            center = vehicle['center']
            vehicle_id = f"{vehicle['vehicle_type']}_{center[0]}_{center[1]}"
            
            if vehicle_id in self.tracked_vehicles:
                speed = self.tracked_vehicles[vehicle_id]['speed_mph']
                vehicle_type = self.tracked_vehicles[vehicle_id]['vehicle_type']
                
                # Adjust speed limit for vehicle type
                effective_speed_limit = speed_limit
                if vehicle_type == 'truck':
                    effective_speed_limit *= truck_penalty
                
                if speed > effective_speed_limit:
                    violation = NYCViolation(
                        timestamp=timestamp,
                        violation_type='speeding',
                        confidence=vehicle['confidence'],
                        location=tuple(center),
                        vehicle_id=vehicle_id,
                        speed_mph=speed,
                        vehicle_type=vehicle_type,
                        zone=self._get_vehicle_zone(center)
                    )
                    violations.append(violation)
        
        return violations
    
    def _check_bike_lane_violations(self, vehicles: List[Dict], timestamp: datetime) -> List[NYCViolation]:
        """Check for vehicles in bike lane"""
        violations = []
        
        for vehicle in vehicles:
            center = vehicle['center']
            zone = self._get_vehicle_zone(center)
            
            # Check if non-bicycle vehicle is in bike lane
            if zone == 'bike_lane' and vehicle['vehicle_type'] != 'bicycle':
                violation = NYCViolation(
                    timestamp=timestamp,
                    violation_type='bike_lane_violation',
                    confidence=vehicle['confidence'],
                    location=tuple(center),
                    vehicle_id=f"{vehicle['vehicle_type']}_{center[0]}_{center[1]}",
                    vehicle_type=vehicle['vehicle_type'],
                    zone=zone
                )
                violations.append(violation)
        
        return violations
    
    def _check_pedestrian_violations(self, vehicles: List[Dict], timestamp: datetime) -> List[NYCViolation]:
        """Check for vehicles blocking pedestrian crossing"""
        violations = []
        
        for vehicle in vehicles:
            center = vehicle['center']
            zone = self._get_vehicle_zone(center)
            
            # Check if vehicle is blocking pedestrian crossing
            if zone == 'pedestrian_crossing':
                violation = NYCViolation(
                    timestamp=timestamp,
                    violation_type='pedestrian_violation',
                    confidence=vehicle['confidence'],
                    location=tuple(center),
                    vehicle_id=f"{vehicle['vehicle_type']}_{center[0]}_{center[1]}",
                    vehicle_type=vehicle['vehicle_type'],
                    zone=zone
                )
                violations.append(violation)
        
        return violations
    
    def _check_bus_lane_violations(self, vehicles: List[Dict], timestamp: datetime) -> List[NYCViolation]:
        """Check for non-bus vehicles in bus lanes"""
        violations = []
        
        # This would need to be configured based on actual bus lane locations
        # For now, we'll check if non-bus vehicles are in areas where buses typically travel
        
        for vehicle in vehicles:
            center = vehicle['center']
            zone = self._get_vehicle_zone(center)
            
            # Check if non-bus vehicle is in bus route area
            if vehicle['vehicle_type'] != 'bus' and zone in ['one_way_avenue_approach', 'one_way_street_approach']:
                # Additional logic could be added here to detect actual bus lanes
                pass
        
        return violations
    
    def _check_expressway_offramp_violations(self, vehicles: List[Dict], timestamp: datetime) -> List[NYCViolation]:
        """Check for violations specific to expressway offramp"""
        violations = []
        
        for vehicle in vehicles:
            center = vehicle['center']
            zone = self._get_vehicle_zone(center)
            
            if zone == 'expressway_offramp':
                vehicle_id = f"{vehicle['vehicle_type']}_{center[0]}_{center[1]}"
                
                if vehicle_id in self.tracked_vehicles:
                    speed = self.tracked_vehicles[vehicle_id]['speed_mph']
                    
                    # Expressway offramps often have lower speed limits
                    if speed > 20:  # Typical offramp speed limit
                        violation = NYCViolation(
                            timestamp=timestamp,
                            violation_type='offramp_speeding',
                            confidence=vehicle['confidence'],
                            location=tuple(center),
                            vehicle_id=vehicle_id,
                            speed_mph=speed,
                            vehicle_type=vehicle['vehicle_type'],
                            zone=zone
                        )
                        violations.append(violation)
        
        return violations
    
    def _point_in_polygon(self, point: List[int], polygon: List[List[int]]) -> bool:
        """Check if point is inside polygon"""
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
    
    def save_violations_to_csv(self, violations: List[NYCViolation], output_path: str):
        """Save NYC violations to CSV with enhanced data"""
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
                'direction': violation.direction,
                'vehicle_type': violation.vehicle_type,
                'zone': violation.zone
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(violations)} NYC violations to {output_path}")


if __name__ == "__main__":
    # Example usage
    analyzer = NYCIntersectionAnalyzer()
    
    # Analyze a video file
    video_path = "data/videos/raw/nyc_intersection_sample.mp4"
    violations = analyzer.analyze_video(video_path)
    
    # Save results
    output_path = "data/outputs/nyc_violations.csv"
    analyzer.save_violations_to_csv(violations, output_path)
    
    print(f"NYC intersection analysis complete. Found {len(violations)} violations.")
