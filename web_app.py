#!/usr/bin/env python3
"""
RedCrowWatch Web Interface

A simple Flask web application for uploading videos and viewing analysis results.
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename
import pandas as pd

# Add src to path
sys.path.append('src')

# Import with error handling for missing modules
try:
    from analysis.integrated_analyzer import IntegratedAnalyzer
    from visualization.traffic_dashboard import TrafficDashboard
    ANALYSIS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Analysis modules not available: {e}")
    ANALYSIS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'redcrowwatch_secret_key'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Configuration
UPLOAD_FOLDER = 'data/videos/raw'
OUTPUT_FOLDER = 'data/outputs'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv'}

# Ensure directories exist
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
Path('templates').mkdir(exist_ok=True)
Path('static').mkdir(exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'RedCrowWatch'})

@app.route('/test')
def test():
    """Test endpoint"""
    return jsonify({
        'message': 'RedCrowWatch is working!',
        'analysis_available': ANALYSIS_AVAILABLE,
        'upload_folder': UPLOAD_FOLDER,
        'output_folder': OUTPUT_FOLDER,
        'allowed_extensions': list(ALLOWED_EXTENSIONS)
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and analysis"""
    try:
        logger.info(f"Upload request received. Files: {list(request.files.keys())}")
        
        if 'file' not in request.files:
            logger.warning("No file in request")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.warning("Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Secure filename and save
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            
            logger.info(f"Saving file: {filename} to {filepath}")
            file.save(filepath)
            
            # Verify file was saved
            if not os.path.exists(filepath):
                logger.error(f"File not saved: {filepath}")
                return jsonify({'error': 'Failed to save file'}), 500
            
            logger.info(f"File uploaded successfully: {filename} ({os.path.getsize(filepath)} bytes)")
            
            # Run analysis
            analysis_id = timestamp
            logger.info(f"Starting analysis with ID: {analysis_id}")
            result = run_analysis(filepath, analysis_id)
            
            if result['success']:
                logger.info(f"Analysis completed successfully: {analysis_id}")
                return jsonify({
                    'success': True,
                    'analysis_id': analysis_id,
                    'message': 'Analysis completed successfully',
                    'results': result['results']
                })
            else:
                logger.error(f"Analysis failed: {result['error']}")
                return jsonify({
                    'success': False,
                    'error': result['error']
                }), 500
        else:
            logger.warning(f"Invalid file type: {file.filename if file else 'None'}")
            return jsonify({'error': 'Invalid file type. Please upload a video file (MP4, AVI, MOV, MKV, WMV)'}), 400
            
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

def run_analysis(video_path, analysis_id):
    """Run the integrated analysis"""
    try:
        logger.info(f"Starting analysis for {video_path}")
        
        if not ANALYSIS_AVAILABLE:
            # Return mock results for demo purposes
            output_dir = os.path.join(OUTPUT_FOLDER, analysis_id)
            os.makedirs(output_dir, exist_ok=True)
            
            # Create mock results
            results = {
                'video_violations': 0,
                'audio_events': 0,
                'correlated_events': 0,
                'traffic_intensity_score': 0.5,
                'safety_score': 0.8,
                'violation_breakdown': {},
                'audio_breakdown': {},
                'has_visualizations': False,
                'message': 'Analysis modules not available - demo mode'
            }
            
            logger.info(f"Demo analysis completed: {analysis_id}")
            return {'success': True, 'results': results}
        
        # Initialize analyzer
        analyzer = IntegratedAnalyzer()
        
        # Run analysis
        analysis_results = analyzer.analyze_video_with_audio(video_path)
        
        # Save results
        output_dir = os.path.join(OUTPUT_FOLDER, analysis_id)
        analyzer.save_integrated_results(analysis_results, output_dir)
        
        # Create visualizations
        dashboard = TrafficDashboard()
        
        # Convert violations to DataFrame for visualization
        video_violations = analysis_results['video_violations']
        if video_violations:
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
            
            # Create visualizations
            dashboard_path = os.path.join(output_dir, 'dashboard.png')
            heatmap_path = os.path.join(output_dir, 'heatmap.png')
            summary_path = os.path.join(output_dir, 'daily_summary.png')
            
            dashboard.create_comprehensive_dashboard(violations_df, dashboard_path)
            dashboard.create_heatmap(violations_df, heatmap_path)
            dashboard.create_daily_summary(violations_df, summary_path)
        else:
            violations_df = pd.DataFrame()
        
        # Prepare results summary
        summary = analysis_results['summary']
        results = {
            'video_violations': len(analysis_results['video_violations']),
            'audio_events': len(analysis_results['audio_events']),
            'correlated_events': len(analysis_results['correlated_events']),
            'traffic_intensity_score': summary['traffic_intensity_score'],
            'safety_score': summary['safety_score'],
            'violation_breakdown': summary['video_analysis']['violations_by_type'],
            'audio_breakdown': summary['audio_analysis']['events_by_type'],
            'has_visualizations': len(violations_df) > 0
        }
        
        logger.info(f"Analysis completed successfully: {analysis_id}")
        return {'success': True, 'results': results}
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return {'success': False, 'error': str(e)}

@app.route('/results/<analysis_id>')
def view_results(analysis_id):
    """View analysis results"""
    try:
        output_dir = os.path.join(OUTPUT_FOLDER, analysis_id)
        
        if not os.path.exists(output_dir):
            return render_template('error.html', message='Analysis not found'), 404
        
        # Load summary data
        summary_file = os.path.join(output_dir, f'analysis_summary_{analysis_id}.csv')
        if os.path.exists(summary_file):
            summary_df = pd.read_csv(summary_file)
            summary = summary_df.iloc[0].to_dict()
        else:
            summary = {}
        
        # Check for visualization files
        dashboard_exists = os.path.exists(os.path.join(output_dir, 'dashboard.png'))
        heatmap_exists = os.path.exists(os.path.join(output_dir, 'heatmap.png'))
        summary_viz_exists = os.path.exists(os.path.join(output_dir, 'daily_summary.png'))
        
        return render_template('results.html', 
                             analysis_id=analysis_id,
                             summary=summary,
                             dashboard_exists=dashboard_exists,
                             heatmap_exists=heatmap_exists,
                             summary_viz_exists=summary_viz_exists)
        
    except Exception as e:
        logger.error(f"Results view error: {e}")
        return render_template('error.html', message=str(e)), 500

@app.route('/download/<analysis_id>/<filename>')
def download_file(analysis_id, filename):
    """Download analysis files"""
    try:
        file_path = os.path.join(OUTPUT_FOLDER, analysis_id, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        logger.error(f"Download error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/results/<analysis_id>')
def api_results(analysis_id):
    """API endpoint for results data"""
    try:
        output_dir = os.path.join(OUTPUT_FOLDER, analysis_id)
        
        if not os.path.exists(output_dir):
            return jsonify({'error': 'Analysis not found'}), 404
        
        # Load all CSV files
        results = {}
        
        # Load video violations
        violations_file = os.path.join(output_dir, f'video_violations_{analysis_id}.csv')
        if os.path.exists(violations_file):
            results['violations'] = pd.read_csv(violations_file).to_dict('records')
        
        # Load audio events
        audio_file = os.path.join(output_dir, f'audio_events_{analysis_id}.csv')
        if os.path.exists(audio_file):
            results['audio_events'] = pd.read_csv(audio_file).to_dict('records')
        
        # Load correlated events
        corr_file = os.path.join(output_dir, f'correlated_events_{analysis_id}.csv')
        if os.path.exists(corr_file):
            results['correlated_events'] = pd.read_csv(corr_file).to_dict('records')
        
        # Load summary
        summary_file = os.path.join(output_dir, f'analysis_summary_{analysis_id}.csv')
        if os.path.exists(summary_file):
            summary_df = pd.read_csv(summary_file)
            results['summary'] = summary_df.iloc[0].to_dict()
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"API results error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    print("🚦 Starting RedCrowWatch Web Interface")
    print("=" * 50)
    print(f"Open your browser and go to: http://localhost:{port}")
    print("=" * 50)
    
    app.run(debug=debug, host='0.0.0.0', port=port)
