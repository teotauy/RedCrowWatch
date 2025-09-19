"""
Traffic Dashboard Visualization Module

This module creates comprehensive visualizations for traffic violation data
including charts, graphs, and summary statistics.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
from pathlib import Path
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrafficDashboard:
    """
    Creates comprehensive traffic violation dashboards and visualizations
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the dashboard with configuration"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Set up matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette(self.config['visualization']['color_palette'])
        
        # Figure settings
        self.fig_size = tuple(self.config['visualization']['figure_size'])
        self.dpi = self.config['visualization']['dpi']
        
        logger.info("TrafficDashboard initialized successfully")
    
    def create_comprehensive_dashboard(self, violations_df: pd.DataFrame, 
                                     output_path: str = "data/outputs/dashboard.png") -> str:
        """
        Create a comprehensive dashboard with multiple charts
        
        Args:
            violations_df: DataFrame containing violation data
            output_path: Path to save the dashboard image
            
        Returns:
            Path to the saved dashboard image
        """
        logger.info("Creating comprehensive traffic dashboard")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Traffic Violation Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Violations by Hour
        self._plot_violations_by_hour(violations_df, axes[0, 0])
        
        # 2. Violations by Type
        self._plot_violations_by_type(violations_df, axes[0, 1])
        
        # 3. Violations by Direction
        self._plot_violations_by_direction(violations_df, axes[0, 2])
        
        # 4. Speed Distribution
        self._plot_speed_distribution(violations_df, axes[1, 0])
        
        # 5. Timeline
        self._plot_violation_timeline(violations_df, axes[1, 1])
        
        # 6. Summary Statistics
        self._plot_summary_stats(violations_df, axes[1, 2])
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Dashboard saved to {output_path}")
        return output_path
    
    def _plot_violations_by_hour(self, df: pd.DataFrame, ax):
        """Plot violations by hour of day"""
        if df.empty:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Violations by Hour')
            return
        
        # Extract hour from timestamp
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        
        # Count violations by hour
        hourly_counts = df.groupby('hour').size()
        
        # Create bar plot
        hourly_counts.plot(kind='bar', ax=ax, color='skyblue', edgecolor='navy', alpha=0.7)
        ax.set_title('Traffic Violations by Hour of Day', fontweight='bold')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Number of Violations')
        ax.set_xticklabels([f'{h:02d}:00' for h in hourly_counts.index], rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(hourly_counts.values):
            ax.text(i, v + 0.1, str(v), ha='center', va='bottom')
    
    def _plot_violations_by_type(self, df: pd.DataFrame, ax):
        """Plot violations by type"""
        if df.empty:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Violations by Type')
            return
        
        # Count violations by type
        type_counts = df['violation_type'].value_counts()
        
        # Create pie chart
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
        wedges, texts, autotexts = ax.pie(type_counts.values, labels=type_counts.index, 
                                         autopct='%1.1f%%', colors=colors[:len(type_counts)])
        
        ax.set_title('Violations by Type', fontweight='bold')
        
        # Improve text readability
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    def _plot_violations_by_direction(self, df: pd.DataFrame, ax):
        """Plot violations by direction"""
        if df.empty:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Violations by Direction')
            return
        
        # Count violations by direction
        direction_counts = df['direction'].value_counts()
        
        # Create horizontal bar plot
        direction_counts.plot(kind='barh', ax=ax, color='lightcoral', edgecolor='darkred', alpha=0.7)
        ax.set_title('Violations by Direction', fontweight='bold')
        ax.set_xlabel('Number of Violations')
        ax.set_ylabel('Direction')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(direction_counts.values):
            ax.text(v + 0.1, i, str(v), va='center')
    
    def _plot_speed_distribution(self, df: pd.DataFrame, ax):
        """Plot speed distribution for speeding violations"""
        if df.empty:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Speed Distribution')
            return
        
        # Filter for speeding violations with speed data
        speeding_df = df[(df['violation_type'] == 'speeding') & (df['speed_mph'].notna())]
        
        if speeding_df.empty:
            ax.text(0.5, 0.5, 'No speeding data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Speed Distribution')
            return
        
        # Create histogram
        ax.hist(speeding_df['speed_mph'], bins=20, color='orange', alpha=0.7, edgecolor='black')
        ax.set_title('Speed Distribution (Speeding Violations)', fontweight='bold')
        ax.set_xlabel('Speed (mph)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Add speed limit line
        speed_limit = self.config['analysis']['violations']['speeding']['speed_limit_mph']
        ax.axvline(speed_limit, color='red', linestyle='--', linewidth=2, label=f'Speed Limit: {speed_limit} mph')
        ax.legend()
    
    def _plot_violation_timeline(self, df: pd.DataFrame, ax):
        """Plot violation timeline"""
        if df.empty:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Violation Timeline')
            return
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'])
        
        # Create timeline plot
        ax.scatter(df['datetime'], range(len(df)), alpha=0.6, s=20, color='purple')
        ax.set_title('Violation Timeline', fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Violation Index')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        ax.grid(True, alpha=0.3)
    
    def _plot_summary_stats(self, df: pd.DataFrame, ax):
        """Plot summary statistics"""
        ax.axis('off')
        
        if df.empty:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            return
        
        # Calculate summary statistics
        total_violations = len(df)
        violation_types = df['violation_type'].value_counts()
        avg_confidence = df['confidence'].mean()
        
        # Get time range
        df['datetime'] = pd.to_datetime(df['timestamp'])
        time_range = f"{df['datetime'].min().strftime('%H:%M')} - {df['datetime'].max().strftime('%H:%M')}"
        
        # Create summary text
        summary_text = f"""
        SUMMARY STATISTICS
        
        Total Violations: {total_violations}
        Analysis Period: {time_range}
        Average Confidence: {avg_confidence:.2f}
        
        Violation Breakdown:
        """
        
        for vtype, count in violation_types.items():
            percentage = (count / total_violations) * 100
            summary_text += f"\n• {vtype.replace('_', ' ').title()}: {count} ({percentage:.1f}%)"
        
        # Add speed statistics if available
        speeding_df = df[df['violation_type'] == 'speeding']
        if not speeding_df.empty and speeding_df['speed_mph'].notna().any():
            avg_speed = speeding_df['speed_mph'].mean()
            max_speed = speeding_df['speed_mph'].max()
            summary_text += f"\n\nSpeed Statistics:"
            summary_text += f"\n• Average Speed: {avg_speed:.1f} mph"
            summary_text += f"\n• Maximum Speed: {max_speed:.1f} mph"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        ax.set_title('Summary Statistics', fontweight='bold')
    
    def create_heatmap(self, violations_df: pd.DataFrame, 
                      output_path: str = "data/outputs/heatmap.png") -> str:
        """
        Create a heatmap showing violation hotspots
        
        Args:
            violations_df: DataFrame containing violation data
            output_path: Path to save the heatmap image
            
        Returns:
            Path to the saved heatmap image
        """
        logger.info("Creating violation heatmap")
        
        if violations_df.empty:
            logger.warning("No violation data available for heatmap")
            return output_path
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create 2D histogram of violation locations
        x = violations_df['location_x']
        y = violations_df['location_y']
        
        # Create heatmap
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
        
        # Plot heatmap
        im = ax.imshow(heatmap.T, origin='lower', cmap='Reds', alpha=0.8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Number of Violations', rotation=270, labelpad=20)
        
        # Customize plot
        ax.set_title('Traffic Violation Heatmap', fontweight='bold', fontsize=14)
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        
        # Add detection zones if available
        self._overlay_detection_zones(ax)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Heatmap saved to {output_path}")
        return output_path
    
    def _overlay_detection_zones(self, ax):
        """Overlay detection zones on the plot"""
        try:
            detection_zones = self.config['analysis']['detection_zones']
            
            for zone in detection_zones:
                coords = np.array(zone['coordinates'])
                polygon = plt.Polygon(coords, fill=False, edgecolor='blue', 
                                    linewidth=2, linestyle='--', alpha=0.7)
                ax.add_patch(polygon)
                
                # Add zone label
                center_x = coords[:, 0].mean()
                center_y = coords[:, 1].mean()
                ax.text(center_x, center_y, zone['name'].replace('_', '\n'), 
                       ha='center', va='center', fontsize=8, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        except Exception as e:
            logger.warning(f"Could not overlay detection zones: {e}")
    
    def create_daily_summary(self, violations_df: pd.DataFrame, 
                           output_path: str = "data/outputs/daily_summary.png") -> str:
        """
        Create a daily summary visualization
        
        Args:
            violations_df: DataFrame containing violation data
            output_path: Path to save the summary image
            
        Returns:
            Path to the saved summary image
        """
        logger.info("Creating daily summary")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if violations_df.empty:
            ax.text(0.5, 0.5, 'No violations detected today', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=16)
            ax.set_title('Daily Traffic Violation Summary', fontweight='bold')
        else:
            # Create a simple summary chart
            violation_counts = violations_df['violation_type'].value_counts()
            
            bars = ax.bar(violation_counts.index, violation_counts.values, 
                         color=['red', 'orange', 'yellow'], alpha=0.7)
            
            ax.set_title('Daily Traffic Violation Summary', fontweight='bold')
            ax.set_ylabel('Number of Violations')
            ax.set_xlabel('Violation Type')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{int(height)}', ha='center', va='bottom')
            
            # Add total count
            total = len(violations_df)
            ax.text(0.02, 0.98, f'Total Violations: {total}', 
                   transform=ax.transAxes, fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Daily summary saved to {output_path}")
        return output_path


if __name__ == "__main__":
    # Example usage
    dashboard = TrafficDashboard()
    
    # Load sample data
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01 08:00:00', periods=50, freq='5min'),
        'violation_type': np.random.choice(['red_light_running', 'speeding', 'wrong_way'], 50),
        'confidence': np.random.uniform(0.7, 0.95, 50),
        'location_x': np.random.randint(100, 800, 50),
        'location_y': np.random.randint(100, 600, 50),
        'speed_mph': np.random.uniform(30, 60, 50),
        'direction': np.random.choice(['northbound', 'southbound', 'eastbound', 'westbound'], 50)
    })
    
    # Create visualizations
    dashboard.create_comprehensive_dashboard(sample_data)
    dashboard.create_heatmap(sample_data)
    dashboard.create_daily_summary(sample_data)
    
    print("Visualizations created successfully!")

