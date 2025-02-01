#!/bin/bash
# Выключаем развернутое
docker-compose down -v

# Генерируем файл servers.json
./generate_servers.sh

# Запускаем контейнеры
docker-compose up -d

# Чистим  артефакты
rm servers.json
