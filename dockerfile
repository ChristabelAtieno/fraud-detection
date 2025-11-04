FROM python:3.10-slim

WORKDIR /app

COPY . /app

# cache dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt 

CMD ["python", "-u" "main.py" ]