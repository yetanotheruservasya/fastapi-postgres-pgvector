#!/bin/bash
# Скрипт генерирует файл servers.json на основе переменных окружения

# Пример переменных, если они не заданы (можно убрать, если гарантировано, что они заданы)
: "${POSTGRES_HOST:=db}"
: "${POSTGRES_USER:=myuser}"
: "${POSTGRES_PASSWORD:=mypassword}"
: "${POSTGRES_DB:=companies}"
: "${ANOTHER_DB:=analytics}"
: "${COMPANIES_SERVER_NAME:=Companies DB}"
: "${ANALYTICS_SERVER_NAME:=Analytics DB}"

cat <<EOF > servers.json
{
  "Servers": {
    "1": {
      "Name": "${COMPANIES_SERVER_NAME}",
      "Group": "Servers",
      "Host": "${POSTGRES_HOST}",
      "Port": 5432,
      "MaintenanceDB": "${POSTGRES_DB}",
      "Username": "${POSTGRES_USER}",
      "Password": "${POSTGRES_PASSWORD}",
      "SSLMode": "prefer"
    },
    "2": {
      "Name": "${ANALYTICS_SERVER_NAME}",
      "Group": "Servers",
      "Host": "${POSTGRES_HOST}",
      "Port": 5432,
      "MaintenanceDB": "${ANOTHER_DB}",
      "Username": "${POSTGRES_USER}",
      "Password": "${POSTGRES_PASSWORD}",
      "SSLMode": "prefer"
    }
  }
}
EOF

echo "servers.json сгенерирован:"
cat servers.json
