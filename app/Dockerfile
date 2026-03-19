# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Install system dependencies before copying files (improves caching)
RUN apt-get update && apt-get install -y --no-install-recommends \
    iputils-ping \
    build-essential \
    libmupdf-dev \
    gcc \
    python3-dev \
    libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*  # Reduce image size
# Copy only requirements.txt first to leverage Docker layer caching


RUN pip install --upgrade pip  


# Install static dependencies first
COPY requirements-static.txt .
RUN pip install -r requirements-static.txt

# Install dynamic dependencies separately
COPY requirements-dynamic.txt .
RUN pip install -r requirements-dynamic.txt


# Copy the rest of the application code
COPY . .

# Expose the application port
EXPOSE 8080

# Use a more production-ready way to start Uvicorn
#CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
# Run app.py when the container launches
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
