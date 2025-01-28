FROM python:3.9-slim

WORKDIR /app

# Install required packages
COPY dependencies.txt .
RUN pip install -r dependencies.txt

# Copy your Python files
COPY maj_server.py .
COPY maj_modele.py .

# Set permissions for the app directory
RUN chmod -R 777 /app

CMD ["python3", "maj_server.py"]
