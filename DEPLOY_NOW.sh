#!/bin/bash
# Quick deployment helper for getting landing page live on redcrowlabs.com
# This script helps with local testing and provides deployment URLs

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=8765

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     RedCrowWatch Landing Page - Deployment Helper         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check which command was requested
case "${1:-test}" in
  test)
    echo "Starting local test server on http://localhost:$PORT"
    echo "Press Ctrl+C to stop"
    echo ""
    cd "$PROJECT_DIR"
    python3 -m http.server $PORT
    ;;

  github)
    echo ""
    echo "📌 GITHUB PAGES DEPLOYMENT"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    echo "1. Go to: https://github.com/teotauy/RedCrowWatch/settings/pages"
    echo ""
    echo "2. Select:"
    echo "   - Source: 'Deploy from a branch'"
    echo "   - Branch: 'main'"
    echo "   - Folder: '/ (root)'"
    echo ""
    echo "3. Click 'Save'"
    echo ""
    echo "4. Site will be available at:"
    echo "   📍 https://teotauy.github.io/RedCrowWatch/"
    echo ""
    echo "5. To use custom domain redcrowlabs.com:"
    echo "   - Go to your domain registrar (GoDaddy, Namecheap, etc.)"
    echo "   - Add CNAME record: redcrowlabs.com → teotauy.github.io"
    echo "   - Or use GitHub's 'Custom domain' setting (easier)"
    echo ""
    echo "⏱️  Takes ~2-5 minutes"
    ;;

  netlify)
    echo ""
    echo "🚀 NETLIFY DEPLOYMENT (Simplest)"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    echo "1. Go to: https://app.netlify.com"
    echo ""
    echo "2. Sign up / Log in (use GitHub for fastest)"
    echo ""
    echo "3. Click 'Add new site' → 'Deploy manually'"
    echo ""
    echo "4. Drag index.html to the upload area"
    echo ""
    echo "5. You'll get a temporary URL like: xxxxxx.netlify.app"
    echo ""
    echo "6. To add your custom domain:"
    echo "   - In Netlify: Site settings → Domain management"
    echo "   - Add: redcrowlabs.com"
    echo "   - Netlify will show you DNS changes to make"
    echo "   - Point nameservers to Netlify's"
    echo ""
    echo "7. Your site live at: https://redcrowlabs.com"
    echo ""
    echo "⏱️  Takes ~3-5 minutes to set up, 24-48 hours for DNS"
    ;;

  verify)
    echo ""
    echo "✓ VERIFICATION CHECKLIST"
    echo "════════════════════════════════════════════════════════════"
    echo ""

    # Check local files
    echo -n "Landing page file exists... "
    if [ -f "$PROJECT_DIR/index.html" ]; then
      echo "✓"
    else
      echo "✗"
    fi

    echo -n "Dashboard link in page... "
    if grep -q "redcrowwatch.onrender.com" "$PROJECT_DIR/index.html"; then
      echo "✓"
    else
      echo "✗"
    fi

    echo -n "Dashboard is live... "
    if curl -s -o /dev/null -w "%{http_code}" https://redcrowwatch.onrender.com/ | grep -q "200"; then
      echo "✓ (HTTP 200)"
    else
      echo "✗ (Check Render app)"
    fi

    echo ""
    echo "After deploying to redcrowlabs.com, test:"
    echo "  1. Visit https://redcrowlabs.com"
    echo "  2. Click 'View Live Dashboard' button"
    echo "  3. Verify dashboard loads with data"
    ;;

  *)
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  test       - Start local test server (default)"
    echo "  github     - Show GitHub Pages deployment steps"
    echo "  netlify    - Show Netlify deployment steps"
    echo "  verify     - Check deployment readiness"
    echo ""
    echo "Example: $0 netlify"
    ;;
esac
