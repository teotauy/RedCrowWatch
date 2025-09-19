"""
Main entry point for RedCrowWatch Phase 1

This script orchestrates the entire Phase 1 workflow:
1. Video analysis
2. Data visualization
3. Social media posting
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from analysis.integrated_analyzer import IntegratedAnalyzer
from visualization.traffic_dashboard import TrafficDashboard
from social.twitter_bot import TwitterBot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/redcrowwatch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run the Phase 1 workflow"""
    parser = argparse.ArgumentParser(description='RedCrowWatch Phase 1 - Traffic Violation Analysis')
    parser.add_argument('--input', '-i', required=True, help='Path to input video file')
    parser.add_argument('--output-dir', '-o', default='data/outputs', help='Output directory for results')
    parser.add_argument('--post-to-twitter', action='store_true', help='Post results to Twitter')
    parser.add_argument('--config', '-c', default='config/config.yaml', help='Configuration file path')
    
    args = parser.parse_args()
    
    logger.info("Starting RedCrowWatch Phase 1 analysis")
    logger.info(f"Input video: {args.input}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Step 1: Analyze video with integrated video + audio analyzer
        logger.info("Step 1: Analyzing video and audio for NYC intersection violations")
        analyzer = IntegratedAnalyzer(args.config)
        analysis_results = analyzer.analyze_video_with_audio(args.input)
        
        # Extract results
        video_violations = analysis_results['video_violations']
        audio_events = analysis_results['audio_events']
        correlated_events = analysis_results['correlated_events']
        summary = analysis_results['summary']
        
        if not video_violations and not audio_events:
            logger.warning("No violations or audio events detected")
            violations_df = pd.DataFrame()
        else:
            # Convert video violations to DataFrame
            violations_data = []
            for violation in video_violations:
                violations_data.append({
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
            violations_df = pd.DataFrame(violations_data)
        
        # Save all results
        analyzer.save_integrated_results(analysis_results, args.output_dir)
        
        # Save violations to CSV (for backward compatibility)
        csv_path = Path(args.output_dir) / f"violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        violations_df.to_csv(csv_path, index=False)
        logger.info(f"Saved {len(video_violations)} video violations to {csv_path}")
        
        # Step 2: Create visualizations
        logger.info("Step 2: Creating visualizations")
        dashboard = TrafficDashboard(args.config)
        
        # Create comprehensive dashboard
        dashboard_path = Path(args.output_dir) / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        dashboard.create_comprehensive_dashboard(violations_df, str(dashboard_path))
        
        # Create heatmap
        heatmap_path = Path(args.output_dir) / f"heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        dashboard.create_heatmap(violations_df, str(heatmap_path))
        
        # Create daily summary
        summary_path = Path(args.output_dir) / f"daily_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        dashboard.create_daily_summary(violations_df, str(summary_path))
        
        logger.info("Visualizations created successfully")
        
        # Step 3: Post to social media (if requested)
        if args.post_to_twitter:
            logger.info("Step 3: Posting to Twitter")
            try:
                twitter_bot = TwitterBot(args.config)
                success = twitter_bot.post_daily_summary(violations_df, str(dashboard_path), str(heatmap_path))
                
                if success:
                    logger.info("Successfully posted to Twitter")
                else:
                    logger.error("Failed to post to Twitter")
                    
            except Exception as e:
                logger.error(f"Twitter posting failed: {e}")
        else:
            logger.info("Skipping Twitter posting (use --post-to-twitter to enable)")
        
        # Summary
        logger.info("=" * 50)
        logger.info("INTEGRATED ANALYSIS COMPLETE")
        logger.info("=" * 50)
        logger.info(f"Video violations detected: {len(video_violations)}")
        logger.info(f"Audio events detected: {len(audio_events)}")
        logger.info(f"Correlated events: {len(correlated_events)}")
        logger.info(f"Traffic intensity score: {summary['traffic_intensity_score']:.1f}/100")
        logger.info(f"Safety score: {summary['safety_score']:.1f}/100")
        logger.info(f"Output files:")
        logger.info(f"  - Video violations CSV: {csv_path}")
        logger.info(f"  - Audio events CSV: data/outputs/audio_events_*.csv")
        logger.info(f"  - Correlated events CSV: data/outputs/correlated_events_*.csv")
        logger.info(f"  - Analysis summary CSV: data/outputs/analysis_summary_*.csv")
        logger.info(f"  - Dashboard: {dashboard_path}")
        logger.info(f"  - Heatmap: {heatmap_path}")
        logger.info(f"  - Daily Summary: {summary_path}")
        
        if video_violations:
            violation_types = violations_df['violation_type'].value_counts()
            logger.info("Video violation breakdown:")
            for vtype, count in violation_types.items():
                logger.info(f"  - {vtype.replace('_', ' ').title()}: {count}")
        
        if audio_events:
            audio_types = summary['audio_analysis']['events_by_type']
            logger.info("Audio event breakdown:")
            for atype, count in audio_types.items():
                logger.info(f"  - {atype.replace('_', ' ').title()}: {count}")
        
        logger.info("RedCrowWatch Integrated Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

