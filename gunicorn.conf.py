# Gunicorn configuration for Railway deployment
import os

# Get port from environment variable, default to 5000
port = os.environ.get('PORT', '5000')
try:
    port = int(port)
except ValueError:
    print(f"Warning: Invalid PORT value '{port}', using default 5000")
    port = 5000

bind = f"0.0.0.0:{port}"
workers = 1
worker_class = "sync"
worker_connections = 1000
timeout = 30
keepalive = 2
max_requests = 1000
max_requests_jitter = 100
preload_app = True
