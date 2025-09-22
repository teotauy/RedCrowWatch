#!/usr/bin/env python3
"""
Audio Analysis Module for RedCrowWatch
Detects horn honks, sirens, and brake squeals in video audio
"""

import os
import logging
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.signal import find_peaks
import pandas as pd

logger = logging.getLogger(__name__)

class AudioAnalyzer:
    """Audio analysis for traffic monitoring"""
    
    def __init__(self, config=None):
        """Initialize audio analyzer"""
        self.config = config or self._get_default_config()
        logger.info("AudioAnalyzer initialized")
    
    def _get_default_config(self):
        """Get default configuration"""
        return {
            'sample_rate': 22050,
            'hop_length': 512,
            'n_fft': 2048,
            'horn_detection': {
                'frequency_min': 200,
                'frequency_max': 2000,
                'duration_min': 0.1,
                'duration_max': 3.0,
                'db_threshold': 40,
                'confidence_threshold': 0.7
            },
            'siren_detection': {
                'frequency_min': 500,
                'frequency_max': 2000,
                'duration_min': 2.0,
                'db_threshold': 35,
                'confidence_threshold': 0.6
            },
            'brake_detection': {
                'frequency_min': 2000,
                'frequency_max': 8000,
                'duration_min': 0.05,
                'duration_max': 0.5,
                'db_threshold': 30,
                'confidence_threshold': 0.5
            }
        }
    
    def extract_audio_from_video(self, video_path):
        """Extract audio from video file"""
        try:
            # Use librosa to load audio
            audio, sr = librosa.load(video_path, sr=self.config['sample_rate'])
            logger.info(f"Extracted audio: {len(audio)} samples at {sr}Hz")
            return audio, sr
        except Exception as e:
            logger.error(f"Failed to extract audio: {e}")
            return None, None
    
    def detect_horn_honks(self, audio, sr):
        """Detect horn honks in audio"""
        try:
            config = self.config['horn_detection']
            
            # Compute spectrogram
            stft = librosa.stft(audio, hop_length=self.config['hop_length'], n_fft=self.config['n_fft'])
            magnitude = np.abs(stft)
            db = librosa.amplitude_to_db(magnitude)
            
            # Focus on horn frequency range
            freq_bins = librosa.fft_frequencies(sr=sr, n_fft=self.config['n_fft'])
            freq_mask = (freq_bins >= config['frequency_min']) & (freq_bins <= config['frequency_max'])
            horn_spectrum = db[freq_mask, :]
            
            # Find peaks in the horn frequency range
            mean_power = np.mean(horn_spectrum, axis=0)
            peaks, _ = find_peaks(mean_power, height=config['db_threshold'])
            
            # Convert peaks to time and filter by duration
            times = librosa.frames_to_time(peaks, sr=sr, hop_length=self.config['hop_length'])
            events = []
            
            for i, time in enumerate(times):
                # Check if this is a sustained sound (not just a spike)
                start_frame = librosa.time_to_frames(time, sr=sr, hop_length=self.config['hop_length'])
                end_frame = min(start_frame + int(config['duration_max'] * sr / self.config['hop_length']), len(mean_power))
                
                if end_frame - start_frame > int(config['duration_min'] * sr / self.config['hop_length']):
                    # Calculate confidence based on power and duration
                    power = mean_power[peaks[i]]
                    duration = (end_frame - start_frame) * self.config['hop_length'] / sr
                    confidence = min(1.0, (power - config['db_threshold']) / 20.0 + duration / config['duration_max'])
                    
                    if confidence >= config['confidence_threshold']:
                        events.append({
                            'timestamp': time,
                            'event_type': 'horn_honk',
                            'confidence': confidence,
                            'power_db': power,
                            'duration': duration
                        })
            
            logger.info(f"Detected {len(events)} horn honks")
            return events
            
        except Exception as e:
            logger.error(f"Horn detection failed: {e}")
            return []
    
    def detect_sirens(self, audio, sr):
        """Detect emergency sirens in audio"""
        try:
            config = self.config['siren_detection']
            
            # Compute spectrogram
            stft = librosa.stft(audio, hop_length=self.config['hop_length'], n_fft=self.config['n_fft'])
            magnitude = np.abs(stft)
            db = librosa.amplitude_to_db(magnitude)
            
            # Focus on siren frequency range
            freq_bins = librosa.fft_frequencies(sr=sr, n_fft=self.config['n_fft'])
            freq_mask = (freq_bins >= config['frequency_min']) & (freq_bins <= config['frequency_max'])
            siren_spectrum = db[freq_mask, :]
            
            # Look for sustained high-power signals
            mean_power = np.mean(siren_spectrum, axis=0)
            sustained_mask = mean_power > config['db_threshold']
            
            # Find continuous segments
            events = []
            in_siren = False
            start_time = 0
            
            for i, is_high in enumerate(sustained_mask):
                time = librosa.frames_to_time(i, sr=sr, hop_length=self.config['hop_length'])
                
                if is_high and not in_siren:
                    start_time = time
                    in_siren = True
                elif not is_high and in_siren:
                    duration = time - start_time
                    if duration >= config['duration_min']:
                        # Calculate confidence
                        segment_power = np.mean(mean_power[max(0, i-int(duration*sr/self.config['hop_length'])):i])
                        confidence = min(1.0, (segment_power - config['db_threshold']) / 15.0)
                        
                        if confidence >= config['confidence_threshold']:
                            events.append({
                                'timestamp': start_time,
                                'event_type': 'siren',
                                'confidence': confidence,
                                'power_db': segment_power,
                                'duration': duration
                            })
                    in_siren = False
            
            logger.info(f"Detected {len(events)} sirens")
            return events
            
        except Exception as e:
            logger.error(f"Siren detection failed: {e}")
            return []
    
    def detect_brake_squeals(self, audio, sr):
        """Detect brake squeals in audio"""
        try:
            config = self.config['brake_detection']
            
            # Compute spectrogram
            stft = librosa.stft(audio, hop_length=self.config['hop_length'], n_fft=self.config['n_fft'])
            magnitude = np.abs(stft)
            db = librosa.amplitude_to_db(magnitude)
            
            # Focus on high-frequency range for brake squeals
            freq_bins = librosa.fft_frequencies(sr=sr, n_fft=self.config['n_fft'])
            freq_mask = (freq_bins >= config['frequency_min']) & (freq_bins <= config['frequency_max'])
            brake_spectrum = db[freq_mask, :]
            
            # Find sharp, high-frequency peaks
            mean_power = np.mean(brake_spectrum, axis=0)
            peaks, _ = find_peaks(mean_power, height=config['db_threshold'])
            
            events = []
            for i, peak in enumerate(peaks):
                time = librosa.frames_to_time(peak, sr=sr, hop_length=self.config['hop_length'])
                
                # Check duration
                start_frame = max(0, peak - int(config['duration_max'] * sr / self.config['hop_length']))
                end_frame = min(peak + int(config['duration_max'] * sr / self.config['hop_length']), len(mean_power))
                duration = (end_frame - start_frame) * self.config['hop_length'] / sr
                
                if config['duration_min'] <= duration <= config['duration_max']:
                    power = mean_power[peak]
                    confidence = min(1.0, (power - config['db_threshold']) / 25.0)
                    
                    if confidence >= config['confidence_threshold']:
                        events.append({
                            'timestamp': time,
                            'event_type': 'brake_squeal',
                            'confidence': confidence,
                            'power_db': power,
                            'duration': duration
                        })
            
            logger.info(f"Detected {len(events)} brake squeals")
            return events
            
        except Exception as e:
            logger.error(f"Brake detection failed: {e}")
            return []
    
    def analyze_audio(self, video_path):
        """Complete audio analysis of video file"""
        try:
            logger.info(f"Starting audio analysis: {video_path}")
            
            # Extract audio
            audio, sr = self.extract_audio_from_video(video_path)
            if audio is None:
                return []
            
            # Detect different audio events
            horn_events = self.detect_horn_honks(audio, sr)
            siren_events = self.detect_sirens(audio, sr)
            brake_events = self.detect_brake_squeals(audio, sr)
            
            # Combine all events
            all_events = horn_events + siren_events + brake_events
            
            # Sort by timestamp
            all_events.sort(key=lambda x: x['timestamp'])
            
            logger.info(f"Audio analysis complete: {len(all_events)} total events")
            return all_events
            
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            return []