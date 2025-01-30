# Базовый образ Python
FROM python:3.12-slim

# Устанавливаем зависимости
RUN apt-get update && apt-get install -y libpq-dev gcc

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы проекта
COPY . .

# Устанавливаем зависимости Python
RUN pip install --no-cache-dir -r requirements.txt

# Запуск FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
