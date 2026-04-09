# Base image optimized for Python and PyTorch
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for FAISS or PyTorch (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from the backend folder
COPY cis-backend/requirements.txt .

# Install dependencies (ignoring cache to save space)
RUN pip install --no-cache-dir -r requirements.txt

# Create cache directory for Hugging Face transformers
ENV TRANSFORMERS_CACHE=/app/cache
RUN mkdir -p /app/cache

# Copy the rest of the backend files
COPY cis-backend/ .

# Expose port (Hugging Face Spaces runs on 7860)
EXPOSE 7860
ENV PORT=7860

# Start the FastAPI server using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
