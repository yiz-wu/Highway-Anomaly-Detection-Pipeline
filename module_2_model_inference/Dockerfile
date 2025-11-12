FROM python:3.8-slim

# 1. System deps (OpenCV, image libs, etc.)
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Set workdir
WORKDIR /app

# 3. Install Python deps
COPY requirements_cv.txt .
RUN pip install --no-cache-dir -r requirements_cv.txt

# 4. Copy your project
COPY mapping ./mapping
COPY models ./models
COPY script ./script

# 5. Default mount point for external models/config
RUN mkdir -p /app/custom_models /app/input /app/output

# 6. Default entrypoint (can be overridden)
ENTRYPOINT ["python", "script/sdf2pixelmap.py"]
