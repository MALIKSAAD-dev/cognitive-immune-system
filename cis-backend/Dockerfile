# Base image optimized for Python and PyTorch
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for FAISS or PyTorch (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies (ignoring cache to save space)
RUN pip install --no-cache-dir -r requirements.txt

# Create cache directory for Hugging Face transformers so it doesn't redownload models on every startup
ENV TRANSFORMERS_CACHE=/app/cache
RUN mkdir -p /app/cache

# Copy the rest of the backend files
COPY . .

# Expose port (Hugging Face Spaces typically routes traffic to port 7860 or you can specify 8000)
# We will use 7860 as it's the default for Gradio/FastAPI on HF Spaces
EXPOSE 7860
ENV PORT=7860

# Start the FastAPI server using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
