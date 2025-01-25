FROM python:3.9
WORKDIR /app
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r dependencies.txt

# Download NLTK data
RUN python3 -m nltk.downloader stopwords wordnet

# Expose Flask port
EXPOSE 5001

CMD ["python3", "maj_server.py"]
