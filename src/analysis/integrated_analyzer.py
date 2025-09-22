#!/usr/bin/env python3
"""
Integrated Analysis Module for RedCrowWatch
Combines video and audio analysis
"""

import os
import logging
import pandas as pd
from datetime import datetime
from .video_analyzer import VideoAnalyzer
from .audio_analyzer import AudioAnalyzer

logger = logging.getLogger(__name__)

class IntegratedAnalyzer:
    """Integrated video and audio analysis"""
    
    def __init__(self, config=None):
        """Initialize integrated analyzer"""
        self.config = config
        self.video_analyzer = VideoAnalyzer(config)
        self.audio_analyzer = AudioAnalyzer(config)
        logger.info("IntegratedAnalyzer initialized")
    
    def analyze_video_with_audio(self, video_path):
        """Run complete analysis on video file"""
        try:
            logger.info(f"Starting integrated analysis: {video_path}")
            
            # Run video analysis
            video_violations = self.video_analyzer.analyze_video(video_path)
            logger.info(f"Video analysis: {len(video_violations)} violations")
            
            # Run audio analysis
            audio_events = self.audio_analyzer.analyze_audio(video_path)
            logger.info(f"Audio analysis: {len(audio_events)} events")
            
            # Correlate audio and video events
            correlated_events = self._correlate_events(video_violations, audio_events)
            logger.info(f"Correlation: {len(correlated_events)} correlated events")
            
            # Generate summary
            summary = self._generate_summary(video_violations, audio_events, correlated_events)
            
            results = {
                'video_violations': video_violations,
                'audio_events': audio_events,
                'correlated_events': correlated_events,
                'summary': summary
            }
            
            logger.info("Integrated analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Integrated analysis failed: {e}")
            raise
    
    def _correlate_events(self, video_violations, audio_events):
        """Correlate audio and video events"""
        try:
            correlated = []
            time_window = 2.0  # 2 second window for correlation
            
            for violation in video_violations:
                violation_time = violation['timestamp']
                
                # Find audio events within time window
                for audio_event in audio_events:
                    audio_time = audio_event['timestamp']
                    time_diff = abs(violation_time - audio_time)
                    
                    if time_diff <= time_window:
                        correlated_event = {
                            'timestamp': violation_time,
                            'video_violation': violation,
                            'audio_event': audio_event,
                            'time_difference': time_diff,
                            'correlation_strength': max(0, 1.0 - time_diff / time_window)
                        }
                        correlated.append(correlated_event)
            
            logger.info(f"Found {len(correlated)} correlated events")
            return correlated
            
        except Exception as e:
            logger.error(f"Event correlation failed: {e}")
            return []
    
    def _generate_summary(self, video_violations, audio_events, correlated_events):
        """Generate analysis summary"""
        try:
            # Count violations by type
            violation_counts = {}
            for violation in video_violations:
                vtype = violation['violation_type']
                violation_counts[vtype] = violation_counts.get(vtype, 0) + 1
            
            # Count audio events by type
            audio_counts = {}
            for event in audio_events:
                etype = event['event_type']
                audio_counts[etype] = audio_counts.get(etype, 0) + 1
            
            # Calculate scores
            total_violations = len(video_violations)
            total_audio_events = len(audio_events)
            
            # Traffic intensity (0-1, higher = more activity)
            traffic_intensity = min(1.0, (total_violations + total_audio_events) / 20.0)
            
            # Safety score (0-1, higher = safer)
            safety_score = max(0.0, 1.0 - (total_violations * 0.1 + total_audio_events * 0.05))
            
            summary = {
                'total_violations': total_violations,
                'total_audio_events': total_audio_events,
                'total_correlated_events': len(correlated_events),
                'traffic_intensity_score': traffic_intensity,
                'safety_score': safety_score,
                'video_analysis': {
                    'violations_by_type': violation_counts,
                    'average_confidence': np.mean([v['confidence'] for v in video_violations]) if video_violations else 0
                },
                'audio_analysis': {
                    'events_by_type': audio_counts,
                    'average_confidence': np.mean([e['confidence'] for e in audio_events]) if audio_events else 0
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return {
                'total_violations': 0,
                'total_audio_events': 0,
                'total_correlated_events': 0,
                'traffic_intensity_score': 0.0,
                'safety_score': 1.0,
                'video_analysis': {'violations_by_type': {}, 'average_confidence': 0},
                'audio_analysis': {'events_by_type': {}, 'average_confidence': 0}
            }
    
    def save_integrated_results(self, results, output_dir):
        """Save analysis results to files"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save video violations
            if results['video_violations']:
                violations_df = pd.DataFrame(results['video_violations'])
                violations_df.to_csv(os.path.join(output_dir, f'video_violations_{os.path.basename(output_dir)}.csv'), index=False)
            
            # Save audio events
            if results['audio_events']:
                audio_df = pd.DataFrame(results['audio_events'])
                audio_df.to_csv(os.path.join(output_dir, f'audio_events_{os.path.basename(output_dir)}.csv'), index=False)
            
            # Save correlated events
            if results['correlated_events']:
                corr_df = pd.DataFrame(results['correlated_events'])
                corr_df.to_csv(os.path.join(output_dir, f'correlated_events_{os.path.basename(output_dir)}.csv'), index=False)
            
            # Save summary
            summary_df = pd.DataFrame([results['summary']])
            summary_df.to_csv(os.path.join(output_dir, f'analysis_summary_{os.path.basename(output_dir)}.csv'), index=False)
            
            logger.info(f"Results saved to: {output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise