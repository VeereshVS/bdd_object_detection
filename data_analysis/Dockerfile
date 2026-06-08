FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create output directory
RUN mkdir -p /output

# Expose Streamlit port
EXPOSE 8501

# Default command runs the analysis script
# Override with dashboard command for interactive mode
CMD ["python", "run_analysis.py", "--data-root", "/data/bdd100k", "--output", "/output"]
