"""
Audio Analysis Module for Horn Honk Detection

This module analyzes audio from video files to detect horn honks,
which can indicate traffic violations, near-misses, and driver frustration.
"""

import librosa
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import logging
import cv2
from dataclasses import dataclass
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AudioEvent:
    """Data class for audio events (horn honks, etc.)"""
    timestamp: datetime
    event_type: str  # 'horn_honk', 'siren', 'brake_squeal', etc.
    confidence: float
    frequency_peak: float
    duration: float
    decibel_level: float
    location: Optional[Tuple[int, int]] = None  # If we can correlate with video


class AudioAnalyzer:
    """
    Analyzes audio from video files to detect traffic-related sounds
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the audio analyzer"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Audio analysis settings
        self.audio_config = self.config.get('audio_analysis', {})
        self.sample_rate = self.audio_config.get('sample_rate', 22050)
        self.hop_length = self.audio_config.get('hop_length', 512)
        self.n_fft = self.audio_config.get('n_fft', 2048)
        
        # Horn detection parameters
        self.horn_config = self.audio_config.get('horn_detection', {})
        self.horn_freq_min = self.horn_config.get('frequency_min', 200)  # Hz
        self.horn_freq_max = self.horn_config.get('frequency_max', 2000)  # Hz
        self.horn_duration_min = self.horn_config.get('duration_min', 0.1)  # seconds
        self.horn_duration_max = self.horn_config.get('duration_max', 3.0)  # seconds
        self.horn_db_threshold = self.horn_config.get('db_threshold', 40)  # dB
        self.horn_confidence_threshold = self.horn_config.get('confidence_threshold', 0.7)
        
        logger.info("AudioAnalyzer initialized successfully")
    
    def analyze_video_audio(self, video_path: str) -> List[AudioEvent]:
        """
        Extract and analyze audio from video file
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of detected audio events
        """
        logger.info(f"Starting audio analysis: {video_path}")
        
        try:
            # Extract audio from video
            audio_data, sr = self._extract_audio_from_video(video_path)
            
            if audio_data is None:
                logger.warning("Could not extract audio from video")
                return []
            
            # Detect horn honks
            horn_events = self._detect_horn_honks(audio_data, sr)
            
            # Detect other traffic sounds
            siren_events = self._detect_sirens(audio_data, sr)
            brake_events = self._detect_brake_squeals(audio_data, sr)
            
            # Combine all events
            all_events = horn_events + siren_events + brake_events
            
            # Sort by timestamp
            all_events.sort(key=lambda x: x.timestamp)
            
            logger.info(f"Audio analysis complete: {len(all_events)} events detected")
            return all_events
            
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            return []
    
    def _extract_audio_from_video(self, video_path: str) -> Tuple[Optional[np.ndarray], int]:
        """Extract audio from video file using OpenCV"""
        try:
            # Use OpenCV to extract audio
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return None, 0
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            
            logger.info(f"Video: {fps} FPS, {frame_count} frames, {duration:.2f}s duration")
            
            # For now, we'll use librosa to extract audio
            # In production, you might want to use ffmpeg for better audio extraction
            audio_data, sr = librosa.load(video_path, sr=self.sample_rate)
            
            cap.release()
            return audio_data, sr
            
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            return None, 0
    
    def _detect_horn_honks(self, audio_data: np.ndarray, sr: int) -> List[AudioEvent]:
        """Detect horn honks in audio data"""
        events = []
        
        try:
            # Convert to frequency domain
            stft = librosa.stft(audio_data, hop_length=self.hop_length, n_fft=self.n_fft)
            magnitude = np.abs(stft)
            frequencies = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
            
            # Find frequency bins for horn range
            freq_mask = (frequencies >= self.horn_freq_min) & (frequencies <= self.horn_freq_max)
            horn_magnitude = magnitude[freq_mask]
            
            # Calculate power in horn frequency range
            horn_power = np.sum(horn_magnitude, axis=0)
            
            # Convert to dB
            horn_db = librosa.power_to_db(horn_power, ref=np.max)
            
            # Find peaks above threshold
            peaks = self._find_audio_peaks(horn_db, self.horn_db_threshold)
            
            # Analyze each peak
            for peak_idx in peaks:
                start_time = peak_idx * self.hop_length / sr
                
                # Find the end of the horn sound
                end_idx = self._find_sound_end(horn_db, peak_idx, self.horn_db_threshold)
                end_time = end_idx * self.hop_length / sr
                
                duration = end_time - start_time
                
                # Check if duration is within horn range
                if self.horn_duration_min <= duration <= self.horn_duration_max:
                    # Calculate confidence based on power and duration
                    peak_power = horn_db[peak_idx]
                    confidence = self._calculate_horn_confidence(peak_power, duration)
                    
                    if confidence >= self.horn_confidence_threshold:
                        # Find peak frequency
                        peak_freq = self._find_peak_frequency(magnitude, peak_idx, frequencies)
                        
                        event = AudioEvent(
                            timestamp=datetime.now() + timedelta(seconds=start_time),
                            event_type='horn_honk',
                            confidence=confidence,
                            frequency_peak=peak_freq,
                            duration=duration,
                            decibel_level=peak_power
                        )
                        events.append(event)
            
            logger.info(f"Detected {len(events)} horn honks")
            
        except Exception as e:
            logger.error(f"Horn detection failed: {e}")
        
        return events
    
    def _detect_sirens(self, audio_data: np.ndarray, sr: int) -> List[AudioEvent]:
        """Detect emergency vehicle sirens"""
        events = []
        
        try:
            # Sirens typically have a sweeping frequency pattern
            stft = librosa.stft(audio_data, hop_length=self.hop_length, n_fft=self.n_fft)
            magnitude = np.abs(stft)
            frequencies = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
            
            # Look for frequency sweeps in siren range (500-2000 Hz)
            siren_freq_mask = (frequencies >= 500) & (frequencies <= 2000)
            siren_magnitude = magnitude[siren_freq_mask]
            
            # Detect sweeping patterns
            # This is a simplified approach - in production you'd want more sophisticated detection
            siren_power = np.sum(siren_magnitude, axis=0)
            siren_db = librosa.power_to_db(siren_power, ref=np.max)
            
            # Look for sustained high-power periods
            sustained_periods = self._find_sustained_audio(siren_db, threshold=35, min_duration=2.0)
            
            for start_idx, end_idx in sustained_periods:
                start_time = start_idx * self.hop_length / sr
                end_time = end_idx * self.hop_length / sr
                duration = end_time - start_time
                
                # Calculate average power and confidence
                avg_power = np.mean(siren_db[start_idx:end_idx])
                confidence = min(avg_power / 50.0, 1.0)  # Normalize confidence
                
                if confidence >= 0.6:
                    event = AudioEvent(
                        timestamp=datetime.now() + timedelta(seconds=start_time),
                        event_type='siren',
                        confidence=confidence,
                        frequency_peak=1000,  # Typical siren frequency
                        duration=duration,
                        decibel_level=avg_power
                    )
                    events.append(event)
            
            logger.info(f"Detected {len(events)} sirens")
            
        except Exception as e:
            logger.error(f"Siren detection failed: {e}")
        
        return events
    
    def _detect_brake_squeals(self, audio_data: np.ndarray, sr: int) -> List[AudioEvent]:
        """Detect brake squeals (high-frequency, short duration)"""
        events = []
        
        try:
            stft = librosa.stft(audio_data, hop_length=self.hop_length, n_fft=self.n_fft)
            magnitude = np.abs(stft)
            frequencies = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
            
            # Brake squeals are typically high frequency (2000-8000 Hz)
            brake_freq_mask = (frequencies >= 2000) & (frequencies <= 8000)
            brake_magnitude = magnitude[brake_freq_mask]
            
            brake_power = np.sum(brake_magnitude, axis=0)
            brake_db = librosa.power_to_db(brake_power, ref=np.max)
            
            # Find short, sharp peaks
            peaks = self._find_audio_peaks(brake_db, threshold=30)
            
            for peak_idx in peaks:
                start_time = peak_idx * self.hop_length / sr
                end_idx = self._find_sound_end(brake_db, peak_idx, threshold=25)
                end_time = end_idx * self.hop_length / sr
                duration = end_time - start_time
                
                # Brake squeals are typically very short (0.1-0.5 seconds)
                if 0.05 <= duration <= 0.5:
                    peak_power = brake_db[peak_idx]
                    confidence = min(peak_power / 40.0, 1.0)
                    
                    if confidence >= 0.5:
                        peak_freq = self._find_peak_frequency(magnitude, peak_idx, frequencies)
                        
                        event = AudioEvent(
                            timestamp=datetime.now() + timedelta(seconds=start_time),
                            event_type='brake_squeal',
                            confidence=confidence,
                            frequency_peak=peak_freq,
                            duration=duration,
                            decibel_level=peak_power
                        )
                        events.append(event)
            
            logger.info(f"Detected {len(events)} brake squeals")
            
        except Exception as e:
            logger.error(f"Brake squeal detection failed: {e}")
        
        return events
    
    def _find_audio_peaks(self, audio_db: np.ndarray, threshold: float) -> List[int]:
        """Find peaks in audio data above threshold"""
        from scipy.signal import find_peaks
        
        # Find peaks with minimum height and distance
        peaks, _ = find_peaks(audio_db, height=threshold, distance=int(0.1 * len(audio_db) / 100))
        return peaks.tolist()
    
    def _find_sound_end(self, audio_db: np.ndarray, start_idx: int, threshold: float) -> int:
        """Find the end of a sound event"""
        # Look for where the signal drops below threshold
        for i in range(start_idx, len(audio_db)):
            if audio_db[i] < threshold:
                return i
        
        # If no drop found, return a reasonable duration
        return min(start_idx + int(0.5 * len(audio_db) / 100), len(audio_db) - 1)
    
    def _find_sustained_audio(self, audio_db: np.ndarray, threshold: float, min_duration: float) -> List[Tuple[int, int]]:
        """Find sustained audio periods above threshold"""
        periods = []
        in_period = False
        start_idx = 0
        
        min_samples = int(min_duration * len(audio_db) / 10)  # Approximate
        
        for i, db in enumerate(audio_db):
            if db >= threshold and not in_period:
                in_period = True
                start_idx = i
            elif db < threshold and in_period:
                in_period = False
                if i - start_idx >= min_samples:
                    periods.append((start_idx, i))
        
        # Handle case where period continues to end
        if in_period and len(audio_db) - start_idx >= min_samples:
            periods.append((start_idx, len(audio_db) - 1))
        
        return periods
    
    def _calculate_horn_confidence(self, peak_power: float, duration: float) -> float:
        """Calculate confidence score for horn detection"""
        # Power confidence (0-1)
        power_conf = min(peak_power / 50.0, 1.0)
        
        # Duration confidence (optimal around 0.5-1.5 seconds)
        if 0.3 <= duration <= 1.5:
            duration_conf = 1.0
        elif 0.1 <= duration < 0.3:
            duration_conf = duration / 0.3
        elif 1.5 < duration <= 3.0:
            duration_conf = 1.0 - (duration - 1.5) / 1.5
        else:
            duration_conf = 0.0
        
        # Combined confidence
        return (power_conf * 0.7 + duration_conf * 0.3)
    
    def _find_peak_frequency(self, magnitude: np.ndarray, time_idx: int, frequencies: np.ndarray) -> float:
        """Find the peak frequency at a given time"""
        if time_idx >= magnitude.shape[1]:
            return 0.0
        
        # Get magnitude spectrum at this time
        spectrum = magnitude[:, time_idx]
        
        # Find peak frequency
        peak_freq_idx = np.argmax(spectrum)
        return frequencies[peak_freq_idx]
    
    def correlate_audio_video_events(self, audio_events: List[AudioEvent], 
                                   video_violations: List, 
                                   time_tolerance: float = 2.0) -> List[Dict]:
        """
        Correlate audio events with video violations
        
        Args:
            audio_events: List of detected audio events
            video_violations: List of video violations
            time_tolerance: Time window for correlation (seconds)
            
        Returns:
            List of correlated events
        """
        correlated_events = []
        
        for audio_event in audio_events:
            for video_violation in video_violations:
                # Calculate time difference
                time_diff = abs((audio_event.timestamp - video_violation.timestamp).total_seconds())
                
                if time_diff <= time_tolerance:
                    correlated_events.append({
                        'audio_event': audio_event,
                        'video_violation': video_violation,
                        'time_difference': time_diff,
                        'correlation_strength': 1.0 - (time_diff / time_tolerance)
                    })
        
        return correlated_events
    
    def save_audio_events_to_csv(self, events: List[AudioEvent], output_path: str):
        """Save audio events to CSV file"""
        data = []
        for event in events:
            data.append({
                'timestamp': event.timestamp,
                'event_type': event.event_type,
                'confidence': event.confidence,
                'frequency_peak': event.frequency_peak,
                'duration': event.duration,
                'decibel_level': event.decibel_level,
                'location_x': event.location[0] if event.location else None,
                'location_y': event.location[1] if event.location else None
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(events)} audio events to {output_path}")


if __name__ == "__main__":
    # Example usage
    analyzer = AudioAnalyzer()
    
    # Analyze audio from video
    video_path = "data/videos/raw/sample_video.mp4"
    events = analyzer.analyze_video_audio(video_path)
    
    # Save results
    output_path = "data/outputs/audio_events.csv"
    analyzer.save_audio_events_to_csv(events, output_path)
    
    print(f"Audio analysis complete. Found {len(events)} events.")
