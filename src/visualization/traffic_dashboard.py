#!/usr/bin/env python3
"""
Traffic Dashboard Visualization Module for RedCrowWatch
Creates visualizations for traffic analysis results
"""

import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class TrafficDashboard:
    """Traffic dashboard visualization"""
    
    def __init__(self, config=None):
        """Initialize dashboard"""
        self.config = config or self._get_default_config()
        plt.style.use('seaborn-v0_8')
        logger.info("TrafficDashboard initialized")
    
    def _get_default_config(self):
        """Get default configuration"""
        return {
            'figure_size': (12, 8),
            'dpi': 100,
            'colors': {
                'primary': '#1f77b4',
                'secondary': '#ff7f0e',
                'success': '#2ca02c',
                'danger': '#d62728',
                'warning': '#ff7f0e',
                'info': '#17a2b8'
            }
        }
    
    def create_comprehensive_dashboard(self, violations_df, output_path):
        """Create comprehensive traffic dashboard"""
        try:
            if violations_df.empty:
                self._create_empty_dashboard(output_path)
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('RedCrowWatch Traffic Analysis Dashboard', fontsize=16, fontweight='bold')
            
            # 1. Violations by type
            self._plot_violations_by_type(violations_df, axes[0, 0])
            
            # 2. Violations over time
            self._plot_violations_over_time(violations_df, axes[0, 1])
            
            # 3. Speed distribution
            self._plot_speed_distribution(violations_df, axes[1, 0])
            
            # 4. Zone analysis
            self._plot_zone_analysis(violations_df, axes[1, 1])
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
            plt.close()
            
            logger.info(f"Dashboard saved: {output_path}")
            
        except Exception as e:
            logger.error(f"Dashboard creation failed: {e}")
            self._create_empty_dashboard(output_path)
    
    def create_heatmap(self, violations_df, output_path):
        """Create violations heatmap"""
        try:
            if violations_df.empty:
                self._create_empty_heatmap(output_path)
                return
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create time-based heatmap
            violations_df['hour'] = pd.to_datetime(violations_df['timestamp'], unit='s').dt.hour
            violations_df['day'] = pd.to_datetime(violations_df['timestamp'], unit='s').dt.day
            
            heatmap_data = violations_df.groupby(['day', 'hour']).size().unstack(fill_value=0)
            
            sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='Reds', ax=ax)
            ax.set_title('Traffic Violations Heatmap (Day vs Hour)')
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel('Day')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
            plt.close()
            
            logger.info(f"Heatmap saved: {output_path}")
            
        except Exception as e:
            logger.error(f"Heatmap creation failed: {e}")
            self._create_empty_heatmap(output_path)
    
    def create_daily_summary(self, violations_df, output_path):
        """Create daily summary visualization"""
        try:
            if violations_df.empty:
                self._create_empty_summary(output_path)
                return
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle('Daily Traffic Summary', fontsize=14, fontweight='bold')
            
            # 1. Total violations
            total_violations = len(violations_df)
            axes[0].bar(['Total Violations'], [total_violations], color=self.config['colors']['danger'])
            axes[0].set_title('Total Violations')
            axes[0].set_ylabel('Count')
            
            # 2. Average speed
            avg_speed = violations_df['speed_mph'].mean() if 'speed_mph' in violations_df.columns else 0
            axes[1].bar(['Average Speed'], [avg_speed], color=self.config['colors']['warning'])
            axes[1].set_title('Average Speed (mph)')
            axes[1].set_ylabel('Speed (mph)')
            
            # 3. Confidence distribution
            avg_confidence = violations_df['confidence'].mean() if 'confidence' in violations_df.columns else 0
            axes[2].bar(['Average Confidence'], [avg_confidence], color=self.config['colors']['info'])
            axes[2].set_title('Average Detection Confidence')
            axes[2].set_ylabel('Confidence')
            axes[2].set_ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
            plt.close()
            
            logger.info(f"Daily summary saved: {output_path}")
            
        except Exception as e:
            logger.error(f"Daily summary creation failed: {e}")
            self._create_empty_summary(output_path)
    
    def _plot_violations_by_type(self, df, ax):
        """Plot violations by type"""
        try:
            if 'violation_type' in df.columns:
                violation_counts = df['violation_type'].value_counts()
                violation_counts.plot(kind='bar', ax=ax, color=self.config['colors']['primary'])
                ax.set_title('Violations by Type')
                ax.set_xlabel('Violation Type')
                ax.set_ylabel('Count')
                ax.tick_params(axis='x', rotation=45)
            else:
                ax.text(0.5, 0.5, 'No violation data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Violations by Type')
        except Exception as e:
            logger.error(f"Violations by type plot failed: {e}")
    
    def _plot_violations_over_time(self, df, ax):
        """Plot violations over time"""
        try:
            if 'timestamp' in df.columns:
                df['time'] = pd.to_datetime(df['timestamp'], unit='s')
                hourly_violations = df.groupby(df['time'].dt.floor('H')).size()
                hourly_violations.plot(ax=ax, color=self.config['colors']['secondary'])
                ax.set_title('Violations Over Time')
                ax.set_xlabel('Time')
                ax.set_ylabel('Violations per Hour')
            else:
                ax.text(0.5, 0.5, 'No timestamp data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Violations Over Time')
        except Exception as e:
            logger.error(f"Violations over time plot failed: {e}")
    
    def _plot_speed_distribution(self, df, ax):
        """Plot speed distribution"""
        try:
            if 'speed_mph' in df.columns:
                ax.hist(df['speed_mph'], bins=20, color=self.config['colors']['success'], alpha=0.7)
                ax.axvline(df['speed_mph'].mean(), color='red', linestyle='--', label=f'Mean: {df["speed_mph"].mean():.1f} mph')
                ax.set_title('Speed Distribution')
                ax.set_xlabel('Speed (mph)')
                ax.set_ylabel('Frequency')
                ax.legend()
            else:
                ax.text(0.5, 0.5, 'No speed data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Speed Distribution')
        except Exception as e:
            logger.error(f"Speed distribution plot failed: {e}")
    
    def _plot_zone_analysis(self, df, ax):
        """Plot zone analysis"""
        try:
            if 'zone' in df.columns:
                zone_counts = df['zone'].value_counts()
                zone_counts.plot(kind='bar', ax=ax, color=self.config['colors']['warning'])
                ax.set_title('Violations by Zone')
                ax.set_xlabel('Detection Zone')
                ax.set_ylabel('Count')
                ax.tick_params(axis='x', rotation=45)
            else:
                ax.text(0.5, 0.5, 'No zone data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Violations by Zone')
        except Exception as e:
            logger.error(f"Zone analysis plot failed: {e}")
    
    def _create_empty_dashboard(self, output_path):
        """Create empty dashboard when no data"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No traffic violations detected\nin this video', 
                ha='center', va='center', fontsize=16, transform=ax.transAxes)
        ax.set_title('RedCrowWatch Traffic Analysis Dashboard')
        ax.axis('off')
        plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
    
    def _create_empty_heatmap(self, output_path):
        """Create empty heatmap when no data"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No data available for heatmap', 
                ha='center', va='center', fontsize=16, transform=ax.transAxes)
        ax.set_title('Traffic Violations Heatmap')
        ax.axis('off')
        plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
    
    def _create_empty_summary(self, output_path):
        """Create empty summary when no data"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No data available for summary', 
                ha='center', va='center', fontsize=16, transform=ax.transAxes)
        ax.set_title('Daily Traffic Summary')
        ax.axis('off')
        plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()