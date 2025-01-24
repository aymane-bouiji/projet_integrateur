# Use Python 3.9 as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY dependencies.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r dependencies.txt

# Copy the application code
COPY maj_server.py .


# Expose port 5001 for the Flask application
EXPOSE 5001

# Command to run the Flask application
CMD ["python", "maj_server.py"]
