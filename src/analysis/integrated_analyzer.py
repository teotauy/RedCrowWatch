"""
Integrated Video + Audio Analyzer for RedCrowWatch

This module combines video analysis and audio analysis to provide
comprehensive traffic monitoring with both visual and audio cues.
"""

import logging
from datetime import datetime
from typing import List, Dict, Tuple
import pandas as pd
from pathlib import Path

from .nyc_intersection_analyzer import NYCIntersectionAnalyzer, NYCViolation
from .audio_analyzer import AudioAnalyzer, AudioEvent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegratedAnalyzer:
    """
    Combines video and audio analysis for comprehensive traffic monitoring
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the integrated analyzer"""
        self.config_path = config_path
        
        # Initialize analyzers
        self.video_analyzer = NYCIntersectionAnalyzer(config_path)
        self.audio_analyzer = AudioAnalyzer(config_path)
        
        logger.info("Integrated Analyzer initialized successfully")
    
    def analyze_video_with_audio(self, video_path: str) -> Dict:
        """
        Perform comprehensive analysis combining video and audio
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing all analysis results
        """
        logger.info(f"Starting integrated analysis: {video_path}")
        
        # Run video analysis
        logger.info("Running video analysis...")
        video_violations = self.video_analyzer.analyze_video(video_path)
        
        # Run audio analysis
        logger.info("Running audio analysis...")
        audio_events = self.audio_analyzer.analyze_video_audio(video_path)
        
        # Correlate audio and video events
        logger.info("Correlating audio and video events...")
        correlated_events = self.audio_analyzer.correlate_audio_video_events(
            audio_events, video_violations
        )
        
        # Generate comprehensive report
        analysis_results = {
            'video_violations': video_violations,
            'audio_events': audio_events,
            'correlated_events': correlated_events,
            'summary': self._generate_analysis_summary(video_violations, audio_events, correlated_events)
        }
        
        logger.info("Integrated analysis complete")
        return analysis_results
    
    def _generate_analysis_summary(self, video_violations: List[NYCViolation], 
                                 audio_events: List[AudioEvent], 
                                 correlated_events: List[Dict]) -> Dict:
        """Generate a comprehensive analysis summary"""
        
        # Video violation summary
        video_summary = {
            'total_violations': len(video_violations),
            'violations_by_type': {},
            'violations_by_zone': {},
            'violations_by_vehicle_type': {}
        }
        
        for violation in video_violations:
            # Count by type
            vtype = violation.violation_type
            video_summary['violations_by_type'][vtype] = video_summary['violations_by_type'].get(vtype, 0) + 1
            
            # Count by zone
            zone = violation.zone or 'unknown'
            video_summary['violations_by_zone'][zone] = video_summary['violations_by_zone'].get(zone, 0) + 1
            
            # Count by vehicle type
            vtype = violation.vehicle_type or 'unknown'
            video_summary['violations_by_vehicle_type'][vtype] = video_summary['violations_by_vehicle_type'].get(vtype, 0) + 1
        
        # Audio event summary
        audio_summary = {
            'total_events': len(audio_events),
            'events_by_type': {},
            'horn_honks': 0,
            'sirens': 0,
            'brake_squeals': 0
        }
        
        for event in audio_events:
            etype = event.event_type
            audio_summary['events_by_type'][etype] = audio_summary['events_by_type'].get(etype, 0) + 1
            
            if etype == 'horn_honk':
                audio_summary['horn_honks'] += 1
            elif etype == 'siren':
                audio_summary['sirens'] += 1
            elif etype == 'brake_squeal':
                audio_summary['brake_squeals'] += 1
        
        # Correlation summary
        correlation_summary = {
            'total_correlations': len(correlated_events),
            'high_confidence_correlations': len([c for c in correlated_events if c['correlation_strength'] > 0.8]),
            'horn_violation_correlations': len([c for c in correlated_events 
                                              if c['audio_event'].event_type == 'horn_honk'])
        }
        
        # Overall summary
        overall_summary = {
            'analysis_timestamp': datetime.now(),
            'video_analysis': video_summary,
            'audio_analysis': audio_summary,
            'correlation_analysis': correlation_summary,
            'traffic_intensity_score': self._calculate_traffic_intensity_score(
                video_violations, audio_events
            ),
            'safety_score': self._calculate_safety_score(
                video_violations, audio_events, correlated_events
            )
        }
        
        return overall_summary
    
    def _calculate_traffic_intensity_score(self, video_violations: List[NYCViolation], 
                                         audio_events: List[AudioEvent]) -> float:
        """Calculate a traffic intensity score (0-100)"""
        # Base score from violations
        violation_score = min(len(video_violations) * 10, 50)  # Max 50 points
        
        # Audio intensity score
        horn_score = min(len([e for e in audio_events if e.event_type == 'horn_honk']) * 5, 30)  # Max 30 points
        siren_score = min(len([e for e in audio_events if e.event_type == 'siren']) * 15, 20)  # Max 20 points
        
        total_score = violation_score + horn_score + siren_score
        return min(total_score, 100)
    
    def _calculate_safety_score(self, video_violations: List[NYCViolation], 
                              audio_events: List[AudioEvent], 
                              correlated_events: List[Dict]) -> float:
        """Calculate a safety score (0-100, higher is safer)"""
        # Start with perfect score
        score = 100.0
        
        # Deduct points for violations
        score -= len(video_violations) * 5
        
        # Deduct points for horn honks (indicates frustration/danger)
        horn_events = [e for e in audio_events if e.event_type == 'horn_honk']
        score -= len(horn_events) * 2
        
        # Deduct points for brake squeals (indicates sudden stops)
        brake_events = [e for e in audio_events if e.event_type == 'brake_squeal']
        score -= len(brake_events) * 3
        
        # Bonus points for sirens (emergency response present)
        siren_events = [e for e in audio_events if e.event_type == 'siren']
        score += len(siren_events) * 2
        
        # Bonus for high correlation (audio matches visual events)
        high_corr = len([c for c in correlated_events if c['correlation_strength'] > 0.8])
        score += high_corr * 1
        
        return max(0, min(score, 100))
    
    def save_integrated_results(self, analysis_results: Dict, output_dir: str):
        """Save all analysis results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save video violations
        if analysis_results['video_violations']:
            video_df = self._violations_to_dataframe(analysis_results['video_violations'])
            video_csv = output_path / f"video_violations_{timestamp}.csv"
            video_df.to_csv(video_csv, index=False)
            logger.info(f"Saved video violations to {video_csv}")
        
        # Save audio events
        if analysis_results['audio_events']:
            audio_df = self._audio_events_to_dataframe(analysis_results['audio_events'])
            audio_csv = output_path / f"audio_events_{timestamp}.csv"
            audio_df.to_csv(audio_csv, index=False)
            logger.info(f"Saved audio events to {audio_csv}")
        
        # Save correlated events
        if analysis_results['correlated_events']:
            corr_df = self._correlated_events_to_dataframe(analysis_results['correlated_events'])
            corr_csv = output_path / f"correlated_events_{timestamp}.csv"
            corr_df.to_csv(corr_csv, index=False)
            logger.info(f"Saved correlated events to {corr_csv}")
        
        # Save summary
        summary_csv = output_path / f"analysis_summary_{timestamp}.csv"
        summary_df = self._summary_to_dataframe(analysis_results['summary'])
        summary_df.to_csv(summary_csv, index=False)
        logger.info(f"Saved analysis summary to {summary_csv}")
    
    def _violations_to_dataframe(self, violations: List[NYCViolation]) -> pd.DataFrame:
        """Convert violations to DataFrame"""
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
        return pd.DataFrame(data)
    
    def _audio_events_to_dataframe(self, events: List[AudioEvent]) -> pd.DataFrame:
        """Convert audio events to DataFrame"""
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
        return pd.DataFrame(data)
    
    def _correlated_events_to_dataframe(self, events: List[Dict]) -> pd.DataFrame:
        """Convert correlated events to DataFrame"""
        data = []
        for event in events:
            data.append({
                'audio_timestamp': event['audio_event'].timestamp,
                'video_timestamp': event['video_violation'].timestamp,
                'time_difference': event['time_difference'],
                'correlation_strength': event['correlation_strength'],
                'audio_event_type': event['audio_event'].event_type,
                'video_violation_type': event['video_violation'].violation_type
            })
        return pd.DataFrame(data)
    
    def _summary_to_dataframe(self, summary: Dict) -> pd.DataFrame:
        """Convert summary to DataFrame"""
        data = [{
            'analysis_timestamp': summary['analysis_timestamp'],
            'total_video_violations': summary['video_analysis']['total_violations'],
            'total_audio_events': summary['audio_analysis']['total_events'],
            'horn_honks': summary['audio_analysis']['horn_honks'],
            'sirens': summary['audio_analysis']['sirens'],
            'brake_squeals': summary['audio_analysis']['brake_squeals'],
            'total_correlations': summary['correlation_analysis']['total_correlations'],
            'traffic_intensity_score': summary['traffic_intensity_score'],
            'safety_score': summary['safety_score']
        }]
        return pd.DataFrame(data)


if __name__ == "__main__":
    # Example usage
    analyzer = IntegratedAnalyzer()
    
    # Analyze video with audio
    video_path = "data/videos/raw/sample_video.mp4"
    results = analyzer.analyze_video_with_audio(video_path)
    
    # Save results
    analyzer.save_integrated_results(results, "data/outputs")
    
    print(f"Integrated analysis complete!")
    print(f"Video violations: {len(results['video_violations'])}")
    print(f"Audio events: {len(results['audio_events'])}")
    print(f"Correlated events: {len(results['correlated_events'])}")
