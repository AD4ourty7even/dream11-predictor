# Use a base Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Create data directory
RUN mkdir -p /app/data

# Copy requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Command to run the application
CMD ["python", "main.py"]