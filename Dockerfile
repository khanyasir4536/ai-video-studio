# AI Video Studio Pro — mobile website ke liye Docker image
FROM python:3.12-slim

# FFmpeg zaroori hai: video assembly (images + voiceover + music) isi se hoti hai
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY app_PREMIUM_v3.py .

# LAN_MODE on: bahar se 0.0.0.0 par suno + password gate (private link)
ENV STUDIO_LAN=1
ENV PYTHONUNBUFFERED=1

EXPOSE 5001

# Render $PORT deta hai; gunicorn single worker (app ka job queue single-process hai)
CMD ["sh", "-c", "gunicorn app_PREMIUM_v3:app --bind 0.0.0.0:${PORT:-5001} --workers 1 --threads 8 --timeout 900"]
