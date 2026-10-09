FROM python:3.11-slim

WORKDIR /app

COPY requirements-serving.txt .

RUN pip install --no-cache-dir -r requirements-serving.txt

COPY . .

ENV MODEL_URI=/app/models/champion_model

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]