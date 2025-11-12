# Use a lightweight Python base
FROM python:3.8-slim

# Set working directory inside container
WORKDIR /app

# Copy project files
COPY script ./script

# Default mount point for config
RUN mkdir -p /app/input /app/output

# Default command: show help
ENTRYPOINT ["python", "script/interpolate_gps.py"]
