#!/bin/bash
# RedCrowWatch Dashboard - run the web dashboard locally

export PATH="/opt/homebrew/bin:$PATH"
cd "$(dirname "$0")"

python3 dashboard.py
