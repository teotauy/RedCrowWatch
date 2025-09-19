# 🚀 Railway Deployment Steps for RedCrowWatch

Follow these steps to deploy your RedCrowWatch system to Railway:

## Step 1: Create GitHub Repository

### Option A: Using GitHub Website
1. Go to [github.com](https://github.com) and sign in
2. Click the **"+"** button → **"New repository"**
3. Repository name: `RedCrowWatch`
4. Description: `AI-Powered Traffic Intersection Monitoring for NYC`
5. Make it **Public** (required for free Railway)
6. **Don't** initialize with README (we already have files)
7. Click **"Create repository"**

### Option B: Using Git Commands (if you have GitHub CLI)
```bash
# Install GitHub CLI first
brew install gh

# Login to GitHub
gh auth login

# Create repository
gh repo create RedCrowWatch --public --description "AI-Powered Traffic Intersection Monitoring for NYC"
```

## Step 2: Push Code to GitHub

```bash
# Add GitHub remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/RedCrowWatch.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 3: Deploy to Railway

### 3.1 Sign Up for Railway
1. Go to [railway.app](https://railway.app)
2. Click **"Login"** → **"Login with GitHub"**
3. Authorize Railway to access your GitHub account

### 3.2 Create New Project
1. Click **"New Project"**
2. Select **"Deploy from GitHub repo"**
3. Find and select your **RedCrowWatch** repository
4. Click **"Deploy Now"**

### 3.3 Configure Deployment
Railway will automatically:
- ✅ Detect it's a Python project
- ✅ Install dependencies from `requirements.txt`
- ✅ Use the `Procfile` for startup
- ✅ Set up the web service

### 3.4 Set Environment Variables
1. Go to your project dashboard
2. Click on your service
3. Go to **"Variables"** tab
4. Add these environment variables:

```
FLASK_ENV=production
PYTHONPATH=/app
```

### 3.5 Configure Domain (Optional)
1. Go to **"Settings"** tab
2. Click **"Generate Domain"** for a custom URL
3. Or use the default Railway domain

## Step 4: Test Your Deployment

1. **Wait for deployment** to complete (2-3 minutes)
2. **Click the generated URL** to access your app
3. **Test the health endpoint**: `https://your-app.railway.app/health`
4. **Upload a test video** to verify full functionality

## Step 5: Monitor and Maintain

### View Logs
- Go to your Railway dashboard
- Click on your service
- View **"Deployments"** tab for logs

### Update Your App
```bash
# Make changes to your code
git add .
git commit -m "Update feature"
git push origin main

# Railway will automatically redeploy!
```

## 🔧 Troubleshooting

### Common Issues:

1. **"Build failed"**
   - Check the build logs in Railway dashboard
   - Ensure all dependencies are in `requirements.txt`
   - Verify Python version compatibility

2. **"App not starting"**
   - Check the runtime logs
   - Verify the `Procfile` is correct
   - Ensure port is set to `$PORT` environment variable

3. **"Module not found"**
   - Add missing dependencies to `requirements.txt`
   - Redeploy the application

4. **"File upload errors"**
   - Check file size limits
   - Verify directory permissions
   - Check available disk space

### Debug Commands:
```bash
# Check local deployment
python3 start_web.py

# Test with Docker
docker-compose up

# Check requirements
pip install -r requirements.txt
```

## 🎉 Success!

Once deployed, you'll have:
- ✅ **Live web application** accessible worldwide
- ✅ **Automatic deployments** from GitHub
- ✅ **Free hosting** with generous limits
- ✅ **Custom domain** option
- ✅ **Easy scaling** when needed

## 📊 Railway Dashboard Features

- **Metrics**: CPU, memory, and network usage
- **Logs**: Real-time application logs
- **Deployments**: Deployment history and status
- **Variables**: Environment variable management
- **Domains**: Custom domain configuration

## 🚀 Next Steps

1. **Share your app** with the generated URL
2. **Set up custom domain** if desired
3. **Monitor usage** in Railway dashboard
4. **Scale up** if you need more resources
5. **Add more features** and redeploy automatically!

---

**Your RedCrowWatch system will be live on Railway in just a few minutes!** 🎉
