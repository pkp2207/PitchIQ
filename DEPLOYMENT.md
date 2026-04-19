# PitchIQ Deployment Guide

This guide covers deploying PitchIQ to **Render** (recommended) or **Vercel**.

---

## Option 1: Deploy to Render (Recommended ✓)

Render is the best choice for Streamlit apps. It natively supports Docker and Python web services.

### Prerequisites
- A Render account ([render.com](https://render.com))
- GitHub repository with PitchIQ code

### Step 1: Prepare Your Repository

1. Push your code to GitHub (if not already done):
   ```bash
   git add .
   git commit -m "Add deployment files"
   git push
   ```

2. Ensure these files are in your repo:
   - `Dockerfile`
   - `render.yaml` (optional, but helpful)
   - `requirements.txt`
   - `pyproject.toml`

### Step 2: Deploy on Render

**Option A: Using render.yaml (Recommended)**

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **New +** → **Blueprint**
3. Connect your GitHub repository
4. Select the repository and branch (`main`)
5. Render will automatically detect `render.yaml` and deploy
6. Wait 5-10 minutes for the build to complete

**Option B: Manual Web Service Deployment**

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **New +** → **Web Service**
3. Connect your GitHub repository
4. Configure:
   - **Name**: `pitchiq`
   - **Environment**: `Docker`
   - **Branch**: `main`
   - **Build Command**: `pip install -e . && pip install -r requirements.txt`
   - **Start Command**: `streamlit run app/streamlit_app.py`
   - **Plan**: `Free` or `Standard`
5. Add Environment Variables (optional):
   ```
   STREAMLIT_SERVER_HEADLESS=true
   STREAMLIT_SERVER_PORT=8501
   ```
6. Click **Deploy**

### Step 3: Monitor Deployment

- Check the deployment logs in the Render dashboard
- Once deployed, you'll get a URL like: `https://pitchiq.onrender.com`
- Visit the URL to access your app

### Step 4: Data & Model Setup (Important!)

By default, Render uses sample data. To use real data:

1. **Option A: Include sample data in repo**
   ```bash
   python scripts/generate_sample_data.py
   python scripts/generate_sample_players.py
   python scripts/generate_sample_sentiment.py
   git add data/raw/
   git commit -m "Add sample data"
   git push
   ```

2. **Option B: Download data at runtime** (Recommended for large datasets)
   - Set up Kaggle API credentials as Render environment variables:
     ```
     KAGGLE_USERNAME=your_username
     KAGGLE_KEY=your_api_key
     ```
   - Update the `Dockerfile` to run data generation:
     ```dockerfile
     RUN python scripts/download_data.py
     ```

3. **Option C: Pre-train models locally, commit to repo**
   ```bash
   python -m src.data.cleaner
   python -m src.features.pipeline
   python -m src.models.train
   git add models/ data/processed/
   git commit -m "Add pre-trained model"
   git push
   ```

### Render Pricing
- **Free tier**: Sleep after 15 minutes of inactivity (fine for demos)
- **Standard**: $7/month (recommended for production)

---

## Option 2: Deploy to Vercel

Vercel is designed for Next.js but can host Streamlit via container deployment.

### Prerequisites
- A Vercel account ([vercel.com](https://vercel.com))
- GitHub repository with PitchIQ code

### Important Note ⚠️
Vercel's free tier has a 12-second timeout for serverless functions, which doesn't work well with Streamlit. **Use Vercel Pro ($20/month) for container deployment**, or prefer **Render** instead.

### Deployment Steps (Vercel Pro with Docker)

1. **Install Vercel CLI**:
   ```bash
   npm install -g vercel
   ```

2. **Log in to Vercel**:
   ```bash
   vercel login
   ```

3. **Deploy**:
   ```bash
   vercel --prod
   ```

4. **Configure in Vercel Dashboard**:
   - Set `Framework` to `Other`
   - Ensure `Dockerfile` is detected
   - Add environment variables as needed
   - Upgrade to Vercel Pro for serverless timeout increase

### Vercel Pricing
- **Free tier**: Limited (12s timeout, not suitable for Streamlit)
- **Pro**: $20/month (suitable for Streamlit containers)

---

## Option 3: Docker Compose (Local Testing)

Before deploying, test locally with Docker:

```bash
# Build the image
docker build -t pitchiq .

# Run the container
docker run -p 8501:8501 pitchiq

# Access at http://localhost:8501
```

---

## Environment Variables

Both platforms support environment variables. Set these if needed:

| Variable | Value | Purpose |
|----------|-------|---------|
| `STREAMLIT_SERVER_HEADLESS` | `true` | Production mode |
| `STREAMLIT_SERVER_PORT` | `8501` | Streamlit port |
| `STREAMLIT_LOGGER_LEVEL` | `info` | Log level |
| `KAGGLE_USERNAME` | Your username | For real data download |
| `KAGGLE_KEY` | Your API key | For real data download |

---

## Troubleshooting

### App Takes Too Long to Load
- **Cause**: Model training runs on every startup
- **Fix**: Pre-train models locally and commit to repo (see Data & Model Setup, Option C)

### "Cannot find module" errors
- **Cause**: Dependencies not installed
- **Fix**: Ensure `requirements.txt` and `pyproject.toml` are up-to-date and committed

### Data files missing
- **Cause**: Raw CSV files not in repository
- **Fix**: Generate sample data or download from Kaggle (see Data & Model Setup)

### Memory/Timeout Issues
- **On Render**: Upgrade to Standard plan
- **On Vercel**: Use Pro plan for Docker containers

---

## Recommended Setup

**For best experience:**
1. Use **Render** with Standard plan ($7/month)
2. Generate sample data with scripts and commit to repo
3. Pre-train models locally and commit `models/` directory
4. Set up GitHub auto-deploy by connecting your repo

This ensures fast startup, no timeouts, and automatic updates on each push.

---

## Additional Resources

- [Render Docs](https://render.com/docs)
- [Vercel Docs](https://vercel.com/docs)
- [Streamlit Cloud Alternative](https://streamlit.io/cloud) - If you want Streamlit's native hosting (simplest option!)
