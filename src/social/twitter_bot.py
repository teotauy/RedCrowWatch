"""
Twitter Bot for Automated Traffic Violation Reporting

This module handles automated posting of traffic violation data to Twitter,
including images, statistics, and engaging content.
"""

import tweepy
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import yaml
from pathlib import Path
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TwitterBot:
    """
    Twitter bot for automated traffic violation reporting
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the Twitter bot with configuration"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Load Twitter API credentials from environment variables
        self.api_key = os.getenv('TWITTER_API_KEY') or self.config['social_media']['twitter']['api_key']
        self.api_secret = os.getenv('TWITTER_API_SECRET') or self.config['social_media']['twitter']['api_secret']
        self.access_token = os.getenv('TWITTER_ACCESS_TOKEN') or self.config['social_media']['twitter']['access_token']
        self.access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET') or self.config['social_media']['twitter']['access_token_secret']
        
        # Initialize Twitter API
        self._initialize_api()
        
        # Get hashtags from config
        self.hashtags = self.config['social_media']['twitter']['hashtags']
        
        logger.info("TwitterBot initialized successfully")
    
    def _initialize_api(self):
        """Initialize Twitter API client"""
        try:
            # Create API client
            self.client = tweepy.Client(
                consumer_key=self.api_key,
                consumer_secret=self.api_secret,
                access_token=self.access_token,
                access_token_secret=self.access_token_secret,
                wait_on_rate_limit=True
            )
            
            # Verify credentials
            user = self.client.get_me()
            logger.info(f"Connected to Twitter as @{user.data.username}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Twitter API: {e}")
            raise
    
    def post_daily_summary(self, violations_df: pd.DataFrame, 
                          dashboard_image_path: str,
                          heatmap_image_path: Optional[str] = None) -> bool:
        """
        Post a daily summary of traffic violations
        
        Args:
            violations_df: DataFrame containing violation data
            dashboard_image_path: Path to the dashboard image
            heatmap_image_path: Optional path to heatmap image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Creating daily summary post")
            
            # Generate tweet content
            tweet_text = self._generate_daily_summary_text(violations_df)
            
            # Prepare media
            media_ids = []
            
            # Upload dashboard image
            if os.path.exists(dashboard_image_path):
                media_id = self._upload_image(dashboard_image_path)
                if media_id:
                    media_ids.append(media_id)
            
            # Upload heatmap if available
            if heatmap_image_path and os.path.exists(heatmap_image_path):
                media_id = self._upload_image(heatmap_image_path)
                if media_id:
                    media_ids.append(media_id)
            
            # Post tweet
            if media_ids:
                response = self.client.create_tweet(
                    text=tweet_text,
                    media_ids=media_ids
                )
            else:
                response = self.client.create_tweet(text=tweet_text)
            
            logger.info(f"Posted daily summary: {response.data['id']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to post daily summary: {e}")
            return False
    
    def post_violation_alert(self, violation_data: Dict) -> bool:
        """
        Post an immediate alert for a serious violation
        
        Args:
            violation_data: Dictionary containing violation information
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Creating violation alert post")
            
            # Generate alert text
            tweet_text = self._generate_violation_alert_text(violation_data)
            
            # Post tweet
            response = self.client.create_tweet(text=tweet_text)
            
            logger.info(f"Posted violation alert: {response.data['id']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to post violation alert: {e}")
            return False
    
    def post_weekly_report(self, weekly_data: pd.DataFrame,
                          weekly_image_path: str) -> bool:
        """
        Post a weekly summary report
        
        Args:
            weekly_data: DataFrame containing weekly violation data
            weekly_image_path: Path to the weekly report image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Creating weekly report post")
            
            # Generate weekly report text
            tweet_text = self._generate_weekly_report_text(weekly_data)
            
            # Upload and post
            if os.path.exists(weekly_image_path):
                media_id = self._upload_image(weekly_image_path)
                if media_id:
                    response = self.client.create_tweet(
                        text=tweet_text,
                        media_ids=[media_id]
                    )
                else:
                    response = self.client.create_tweet(text=tweet_text)
            else:
                response = self.client.create_tweet(text=tweet_text)
            
            logger.info(f"Posted weekly report: {response.data['id']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to post weekly report: {e}")
            return False
    
    def _generate_daily_summary_text(self, violations_df: pd.DataFrame) -> str:
        """Generate text for daily summary tweet"""
        if violations_df.empty:
            return f"🚦 Daily Traffic Report - No violations detected today! 🎉\n\nOur intersection monitoring system is working perfectly. Stay safe out there! {self._get_hashtags()}"
        
        # Calculate statistics
        total_violations = len(violations_df)
        violation_types = violations_df['violation_type'].value_counts()
        
        # Get most common violation type
        most_common = violation_types.index[0] if not violation_types.empty else "unknown"
        most_common_count = violation_types.iloc[0] if not violation_types.empty else 0
        
        # Get time range
        violations_df['datetime'] = pd.to_datetime(violations_df['timestamp'])
        time_range = f"{violations_df['datetime'].min().strftime('%H:%M')} - {violations_df['datetime'].max().strftime('%H:%M')}"
        
        # Generate engaging text
        emoji_map = {
            'red_light_running': '🔴',
            'speeding': '⚡',
            'wrong_way': '↩️'
        }
        
        emoji = emoji_map.get(most_common, '🚗')
        
        # Create tweet text
        tweet_text = f"🚦 Daily Traffic Report - {total_violations} violations detected\n\n"
        tweet_text += f"📊 Most common: {emoji} {most_common.replace('_', ' ').title()} ({most_common_count} cases)\n"
        tweet_text += f"⏰ Analysis period: {time_range}\n\n"
        
        # Add educational message
        educational_messages = [
            "Remember: Red lights save lives! 🛑",
            "Slow down and stay safe! 🚗💨",
            "Traffic safety is everyone's responsibility! 👥",
            "Let's work together for safer streets! 🤝",
            "Every violation puts lives at risk! ⚠️"
        ]
        
        tweet_text += f"{random.choice(educational_messages)}\n\n"
        tweet_text += self._get_hashtags()
        
        return tweet_text
    
    def _generate_violation_alert_text(self, violation_data: Dict) -> str:
        """Generate text for immediate violation alert"""
        violation_type = violation_data.get('violation_type', 'unknown')
        timestamp = violation_data.get('timestamp', datetime.now())
        speed = violation_data.get('speed_mph')
        
        emoji_map = {
            'red_light_running': '🔴',
            'speeding': '⚡',
            'wrong_way': '↩️'
        }
        
        emoji = emoji_map.get(violation_type, '🚗')
        
        tweet_text = f"⚠️ TRAFFIC VIOLATION ALERT ⚠️\n\n"
        tweet_text += f"{emoji} {violation_type.replace('_', ' ').title()} detected\n"
        tweet_text += f"🕐 Time: {timestamp.strftime('%H:%M:%S')}\n"
        
        if speed:
            tweet_text += f"⚡ Speed: {speed:.1f} mph\n"
        
        tweet_text += f"\nStay alert and drive safely! {self._get_hashtags()}"
        
        return tweet_text
    
    def _generate_weekly_report_text(self, weekly_data: pd.DataFrame) -> str:
        """Generate text for weekly report tweet"""
        if weekly_data.empty:
            return f"📊 Weekly Traffic Report - No violations this week! 🎉\n\nOur intersection is getting safer! Keep up the good work! {self._get_hashtags()}"
        
        # Calculate weekly statistics
        total_violations = len(weekly_data)
        daily_average = total_violations / 7
        
        # Get trend information
        weekly_data['date'] = pd.to_datetime(weekly_data['timestamp']).dt.date
        daily_counts = weekly_data.groupby('date').size()
        
        # Find busiest day
        busiest_day = daily_counts.idxmax()
        busiest_count = daily_counts.max()
        
        tweet_text = f"📊 Weekly Traffic Report\n\n"
        tweet_text += f"📈 Total violations: {total_violations}\n"
        tweet_text += f"📅 Daily average: {daily_average:.1f}\n"
        tweet_text += f"🔥 Busiest day: {busiest_day} ({busiest_count} violations)\n\n"
        
        # Add trend analysis
        if daily_average > 10:
            tweet_text += "⚠️ High violation rate detected. Let's work together for safer streets!\n\n"
        elif daily_average < 5:
            tweet_text += "✅ Great job! Low violation rate this week. Keep it up!\n\n"
        else:
            tweet_text += "📊 Moderate violation rate. Room for improvement!\n\n"
        
        tweet_text += self._get_hashtags()
        
        return tweet_text
    
    def _upload_image(self, image_path: str) -> Optional[str]:
        """Upload image to Twitter and return media ID"""
        try:
            # Note: This requires the v1.1 API for media upload
            # You'll need to set up both v2 and v1.1 API clients
            logger.warning("Image upload requires v1.1 API setup - returning None")
            return None
            
        except Exception as e:
            logger.error(f"Failed to upload image {image_path}: {e}")
            return None
    
    def _get_hashtags(self) -> str:
        """Get formatted hashtags string"""
        return " ".join(self.hashtags)
    
    def post_educational_content(self) -> bool:
        """Post educational content about traffic safety"""
        try:
            educational_tweets = [
                "🚦 Did you know? Red light running causes over 800 deaths annually in the US. Always stop on red! #TrafficSafety #RedLightViolations",
                "⚡ Speed kills! Every 10 mph over the speed limit doubles your risk of a fatal crash. Slow down and save lives! #TrafficSafety #Speeding",
                "🔄 Wrong-way driving is often fatal. Always pay attention to road signs and markings! #TrafficSafety #WrongWay",
                "👥 Traffic safety is everyone's responsibility. Drivers, pedestrians, and cyclists - we're all in this together! #TrafficSafety #Community",
                "📱 Distracted driving is dangerous driving. Put your phone away and focus on the road! #TrafficSafety #DistractedDriving"
            ]
            
            tweet_text = random.choice(educational_tweets)
            response = self.client.create_tweet(text=tweet_text)
            
            logger.info(f"Posted educational content: {response.data['id']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to post educational content: {e}")
            return False
    
    def reply_to_mentions(self) -> bool:
        """Reply to mentions and direct messages"""
        try:
            # Get recent mentions
            mentions = self.client.get_mentions()
            
            for mention in mentions.data or []:
                # Generate appropriate response
                response_text = self._generate_mention_response(mention)
                
                # Reply to mention
                self.client.create_tweet(
                    text=response_text,
                    in_reply_to_tweet_id=mention.id
                )
                
                logger.info(f"Replied to mention: {mention.id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to reply to mentions: {e}")
            return False
    
    def _generate_mention_response(self, mention) -> str:
        """Generate response to mention"""
        responses = [
            "Thanks for your interest in traffic safety! 🚦 Our system monitors intersections to help make streets safer for everyone. #TrafficSafety",
            "Great question! Our AI-powered system detects red light running, speeding, and other violations to promote safer driving. 🚗💨",
            "We're working to make our community safer through data-driven traffic monitoring! 📊 #CommunitySafety",
            "Thanks for following our traffic safety updates! Together we can reduce violations and save lives. 🤝 #TrafficSafety"
        ]
        
        return random.choice(responses)


if __name__ == "__main__":
    # Example usage
    bot = TwitterBot()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01 08:00:00', periods=25, freq='10min'),
        'violation_type': ['red_light_running'] * 15 + ['speeding'] * 8 + ['wrong_way'] * 2,
        'confidence': [0.85] * 25,
        'location_x': [400] * 25,
        'location_y': [300] * 25,
        'speed_mph': [35] * 25
    })
    
    # Post daily summary (this will fail without proper API credentials)
    # success = bot.post_daily_summary(sample_data, "data/outputs/dashboard.png")
    # print(f"Post successful: {success}")
    
    print("Twitter bot initialized successfully!")

