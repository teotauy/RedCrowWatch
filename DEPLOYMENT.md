# Deploying RedCrowWatch Landing Page

The landing page (`index.html` / `landing.html`) is a static HTML file that serves as the entry point for the RedCrowWatch project. It links to the live dashboard at **https://redcrowwatch.onrender.com**.

## Quick Deploy Options

### Option 1: GitHub Pages (Simplest)

If you own the GitHub repository and want to use GitHub Pages:

1. Go to your repo settings: https://github.com/teotauy/RedCrowWatch/settings/pages
2. Select **Deploy from a branch**
3. Choose branch: **main**, folder: **/ (root)**
4. Click **Save**

Your landing page will be available at `https://teotauy.github.io/RedCrowWatch/`

To point `redcrowlabs.com` to this GitHub Pages site:
- Add a CNAME record in your domain registrar pointing to `teotauy.github.io`
- OR add the domain in GitHub Pages settings (requires DNS CNAME verification)

---

### Option 2: Netlify (Free, Easiest with Custom Domain)

1. Go to https://netlify.com and sign up
2. Click **Add new site** → **Deploy manually**
3. Drag `index.html` into the upload area
4. Your site gets a random Netlify URL
5. In Netlify settings, add your custom domain `redcrowlabs.com`
6. Point your domain's nameservers to Netlify's (instructions provided in Netlify dashboard)

---

### Option 3: Vercel

1. Go to https://vercel.com and sign up
2. Click **New Project** → **Import Git Repository**
3. Select the RedCrowWatch repo
4. In settings, add `redcrowlabs.com` as a custom domain
5. Point domain nameservers to Vercel

---

### Option 4: Manual Upload (If Already Hosting redcrowlabs.com)

If you already have a hosting provider and FTP/SSH access:

```bash
# Copy the landing page to your web root
scp index.html user@your-host:/path/to/public_html/
```

Or via FTP, upload `index.html` to your web root as the default document.

---

### Option 5: Simple Python Server (Local Testing)

To test the landing page locally:

```bash
cd /Users/colbyblack/RedCrowWatch
python3 -m http.server 8080
```

Then visit: http://localhost:8080/

---

## File Locations

The landing page is available in two locations (they're identical):
- `index.html` — Root of repo (standard for static hosting)
- `landing.html` — Explicitly named version

Most static hosting services will automatically serve `index.html` as the default document, so you only need to deploy `index.html`.

---

## Current Live Dashboard

The landing page links to the live dashboard at:
**https://redcrowwatch.onrender.com**

Make sure this URL is accessible before promoting the landing page link to DOT. The dashboard displays:
- Real-time violation statistics
- Chart visualizations by violation type and location
- Recent violation timeline
- CSV export capability
- Auto-refresh every 30 seconds

---

## Next Steps

1. Choose your preferred deployment option above
2. Deploy `index.html` to `redcrowlabs.com`
3. Test the landing page by visiting `https://redcrowlabs.com`
4. Verify the "View Live Dashboard" button links correctly to https://redcrowwatch.onrender.com
5. Share `https://redcrowlabs.com` with DOT

---

## Troubleshooting

**"404 Not Found" after deploying:**
- Ensure the file is named `index.html` (not `landing.html`)
- Check that it's in the web root directory
- Verify your hosting service's default document setting

**"Cannot reach Render dashboard":**
- The Render app may have gone to sleep (free tier limitation)
- Visit https://redcrowwatch.onrender.com directly to wake it up
- Check Render dashboard at https://render.com/dashboard

**DNS not resolving:**
- Changes can take 24-48 hours to propagate
- Flush your local DNS cache:
  ```bash
  sudo dscacheutil -flushcache  # macOS
  ```
- Check propagation: https://mxtoolbox.com/dnslookupp/
