FROM python:3.10.12-slim

WORKDIR /app

RUN pip install --upgrade pip
RUN apt-get update && apt-get install -y p7zip-full
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN 7z x db.7z

CMD ["python", "bot.py"]
