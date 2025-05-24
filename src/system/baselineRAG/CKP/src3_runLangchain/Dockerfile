FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "main.py"] 