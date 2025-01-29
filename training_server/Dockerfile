FROM python:3.9-slim

WORKDIR /app


COPY dependencies.txt .
RUN pip install -r dependencies.txt


COPY maj_server.py .
COPY maj_modele.py .

EXPOSE 5011

RUN chmod -R 777 /app

CMD ["python3", "maj_server.py"]
