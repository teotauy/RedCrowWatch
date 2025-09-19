# RedCrowWatch Deployment Guide

This guide covers multiple deployment options for your RedCrowWatch traffic monitoring system.

## 🚀 Quick Deploy Options

### 1. 🆓 **Railway (Recommended - FREE)**
**Best for: Quick deployment, free tier, automatic deploys**

#### Steps:
1. **Sign up** at [railway.app](https://railway.app)
2. **Connect GitHub** repository
3. **Deploy** automatically from your repo
4. **Set environment variables** in Railway dashboard
5. **Access** your live app!

#### Environment Variables:
```
FLASK_ENV=production
PYTHONPATH=/app
```

#### Benefits:
- ✅ Free tier available
- ✅ Automatic deploys from GitHub
- ✅ Built-in database support
- ✅ Easy scaling

---

### 2. 🆓 **Render (FREE)**
**Best for: Simple deployment, good free tier**

#### Steps:
1. **Sign up** at [render.com](https://render.com)
2. **Connect GitHub** repository
3. **Create Web Service** from your repo
4. **Configure** build settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn web_app:app`
5. **Deploy** and access your app!

#### Benefits:
- ✅ Free tier with 750 hours/month
- ✅ Automatic SSL certificates
- ✅ Easy GitHub integration

---

### 3. 🆓 **PythonAnywhere (FREE)**
**Best for: Python-focused hosting**

#### Steps:
1. **Sign up** at [pythonanywhere.com](https://pythonanywhere.com)
2. **Upload** your code via Git or file upload
3. **Create Web App** with Flask
4. **Configure** WSGI file
5. **Deploy** and access!

#### Benefits:
- ✅ Free tier available
- ✅ Python-optimized
- ✅ Easy file management

---

### 4. 🐳 **Docker Deployment (Any VPS)**
**Best for: Full control, any cloud provider**

#### Steps:
1. **Clone** repository on your server
2. **Install Docker** and Docker Compose
3. **Run**: `docker-compose up -d`
4. **Access** your app on port 5000

#### Commands:
```bash
# Clone repo
git clone https://github.com/yourusername/RedCrowWatch.git
cd RedCrowWatch

# Start with Docker
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

### 5. ☁️ **Cloud Providers (PAID)**
**Best for: Production, high traffic, full control**

#### AWS EC2:
- Launch Ubuntu 22.04 instance
- Install Docker and Docker Compose
- Deploy with Docker
- Configure security groups for port 5000

#### Google Cloud Run:
- Build and push Docker image
- Deploy to Cloud Run
- Configure environment variables
- Set up custom domain

#### DigitalOcean:
- Create Droplet (Ubuntu 22.04)
- Install Docker and Docker Compose
- Deploy with Docker
- Configure firewall

---

## 🔧 Configuration for Production

### Environment Variables:
```bash
# Required
FLASK_ENV=production
PYTHONPATH=/app

# Optional
TWITTER_API_KEY=your_key_here
TWITTER_API_SECRET=your_secret_here
TWITTER_ACCESS_TOKEN=your_token_here
TWITTER_ACCESS_TOKEN_SECRET=your_secret_here
```

### File Permissions:
```bash
# Make scripts executable
chmod +x start_web.py
chmod +x scripts/setup_phase1.py

# Create directories
mkdir -p data/videos/raw data/outputs logs
```

### Security Considerations:
- Use environment variables for secrets
- Enable HTTPS in production
- Set up proper file upload limits
- Configure firewall rules
- Use a reverse proxy (nginx) for production

---

## 📊 Static Site Alternative (GitHub Pages)

Since GitHub Pages only supports static sites, here's a workaround:

### Option A: Frontend-Only Demo
1. **Create static HTML** version of the interface
2. **Use mock data** for demonstrations
3. **Host on GitHub Pages** for showcasing
4. **Link to live app** for actual analysis

### Option B: Hybrid Approach
1. **Host frontend** on GitHub Pages
2. **Host backend** on Railway/Render
3. **Connect** via API calls
4. **Best of both worlds**

---

## 🚀 Recommended Deployment Strategy

### For Development/Testing:
- **Local**: `python3 start_web.py`
- **Docker**: `docker-compose up`

### For Production:
1. **Railway** (free tier) - Best overall
2. **Render** (free tier) - Good alternative
3. **Docker on VPS** - Full control

### For Showcase:
- **GitHub Pages** - Static demo site
- **Live app** - Link to actual deployment

---

## 🔍 Troubleshooting

### Common Issues:

1. **"Module not found" errors**
   - Check requirements.txt
   - Verify Python version
   - Reinstall dependencies

2. **"Port already in use" errors**
   - Change port in web_app.py
   - Kill existing processes
   - Use different port

3. **"File upload failed" errors**
   - Check file size limits
   - Verify file permissions
   - Check disk space

4. **"Memory errors" during analysis**
   - Reduce video file size
   - Increase server memory
   - Use smaller video resolution

### Debug Mode:
```python
# In web_app.py, change:
app.run(debug=True, host='0.0.0.0', port=5000)
```

---

## 📈 Scaling Considerations

### For High Traffic:
- Use **Gunicorn** with multiple workers
- Implement **Redis** for caching
- Use **CDN** for static files
- Set up **load balancing**

### For Large Files:
- Increase **upload limits**
- Use **cloud storage** (S3, GCS)
- Implement **chunked uploads**
- Add **progress tracking**

---

## 🎯 Next Steps

1. **Choose deployment option** based on your needs
2. **Set up repository** on GitHub
3. **Deploy** using chosen method
4. **Configure** environment variables
5. **Test** with sample video
6. **Share** your live app!

---

**Ready to deploy? Pick your platform and follow the steps above!** 🚀
