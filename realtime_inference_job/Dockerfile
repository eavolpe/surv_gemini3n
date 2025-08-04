# Use lightweight Python image
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy files
COPY . ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Entry point
CMD ["python", "job.py"]
