FROM python:3.13-slim

WORKDIR /app

COPY backend_requirements.txt .
RUN pip install --no-cache-dir -r backend_requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]