#!/usr/bin/env python3
"""
Simple startup script for Railway deployment
"""

import os
import sys

if __name__ == '__main__':
    # Get port from environment variable, default to 5000
    port_str = os.environ.get('PORT', '5000')
    try:
        port = int(port_str)
    except (ValueError, TypeError):
        print(f"Warning: Invalid PORT value '{port_str}', using default 5000")
        port = 5000
    
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    print("🚦 Starting RedCrowWatch Web Interface")
    print("=" * 50)
    print(f"Port: {port}")
    print(f"Debug: {debug}")
    print(f"Environment: {os.environ.get('FLASK_ENV', 'development')}")
    print("=" * 50)
    
    # Import app after setting up environment
    try:
        from web_app import app
        print("✅ Flask app imported successfully")
        app.run(debug=debug, host='0.0.0.0', port=port)
    except Exception as e:
        print(f"❌ Error starting Flask app: {e}")
        sys.exit(1)
