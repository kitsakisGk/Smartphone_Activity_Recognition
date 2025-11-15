# 🐳 How to Run Docker on Windows

## Step 1: Install Docker Desktop

1. **Download Docker Desktop for Windows**
   - Go to: https://www.docker.com/products/docker-desktop/
   - Click "Download for Windows"
   - Run the installer

2. **Start Docker Desktop**
   - Open Docker Desktop from Start Menu
   - Wait for it to say "Docker Desktop is running"
   - You'll see a whale icon in your system tray

## Step 2: Run Your Project with Docker

### Option A: Using Docker Compose (Recommended - Easiest!)

1. **Open Command Prompt (CMD)**
   ```cmd
   Press Win + R
   Type: cmd
   Press Enter
   ```

2. **Navigate to your project**
   ```cmd
   cd D:\Ptyxiakhh\Smartphone_Activity_Recognition
   ```

3. **Run Docker Compose**
   ```cmd
   docker-compose up --build
   ```

4. **Access the app**
   - Wait for "You can now view your Streamlit app in your browser"
   - Open browser: http://localhost:8501

5. **Stop the app**
   ```cmd
   Press Ctrl + C in the CMD window
   ```

### Option B: Using Docker Commands Manually

1. **Build the image**
   ```cmd
   cd D:\Ptyxiakhh\Smartphone_Activity_Recognition
   docker build -t activity-recognition .
   ```

2. **Run the container**
   ```cmd
   docker run -p 8501:8501 activity-recognition
   ```

3. **Access the app**
   - Open browser: http://localhost:8501

## Troubleshooting

### Problem: "docker: command not found"
**Solution**: Docker Desktop is not installed or not running
- Install Docker Desktop
- Start Docker Desktop and wait for it to fully start

### Problem: "Error response from daemon: driver failed"
**Solution**: Enable WSL 2 (Windows Subsystem for Linux)
1. Open PowerShell as Administrator
2. Run: `wsl --install`
3. Restart computer
4. Start Docker Desktop again

### Problem: Port 8501 is already in use
**Solution**: Another Streamlit app is running
```cmd
# Find what's using port 8501
netstat -ano | findstr :8501

# Kill the process (replace PID with the number from above)
taskkill /PID <PID> /F
```

### Problem: "Cannot connect to Docker daemon"
**Solution**: Docker Desktop is not running
- Open Docker Desktop from Start Menu
- Wait for it to say "running"

## What Docker Does

Think of Docker like a **portable computer** that contains:
- ✅ Python 3.10
- ✅ All libraries (TensorFlow, scikit-learn, Streamlit)
- ✅ Your code
- ✅ Everything needed to run

**Benefits**:
- Works the same on Windows, Mac, Linux
- Easy to deploy to cloud (AWS, Google Cloud, Azure)
- No "it works on my machine" problems

## Quick Reference

```cmd
# Build and run (one command)
docker-compose up --build

# Run in background (detached mode)
docker-compose up -d

# Stop containers
docker-compose down

# View running containers
docker ps

# View logs
docker logs smartphone-activity-recognition

# Remove all containers and images (clean slate)
docker system prune -a
```

## Next Steps

Once you verify Docker works locally:
1. **Deploy to Heroku**: Free tier for demos
2. **Deploy to AWS**: More scalable, ~$20/month
3. **Deploy to Google Cloud Run**: Pay-per-request

See [docs/PRODUCTION_DEPLOYMENT.md](docs/PRODUCTION_DEPLOYMENT.md) for cloud deployment guides!
