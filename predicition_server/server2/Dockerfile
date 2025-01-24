# Use Python 3.9 as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY dependencies.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r dependencies.txt

# Copy the application code (excluding the model files)
COPY server.py .

# Expose port 5000 for the Flask application
EXPOSE 5000

# Command to run the Flask application
CMD ["python", "server.py"]
