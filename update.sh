#!/bin/bash
# Выключаем развернутое
docker-compose down -v

# Генерируем файл servers.json
./generate_servers.sh

# Пересобираем образы
docker-compose build

# Запускаем контейнеры
docker-compose up -d

# Чистим  артефакты
rm servers.json
