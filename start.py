#!/usr/bin/env python3
"""
Simple startup script for Railway deployment
"""

import os
import sys
from web_app import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    print("🚦 Starting RedCrowWatch Web Interface")
    print("=" * 50)
    print(f"Port: {port}")
    print(f"Debug: {debug}")
    print("=" * 50)
    
    app.run(debug=debug, host='0.0.0.0', port=port)
