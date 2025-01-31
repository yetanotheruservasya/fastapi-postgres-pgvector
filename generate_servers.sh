#!/bin/bash

# Проверяем, что переменные окружения установлены
if [[ -z "$POSTGRES_DB" || -z "$POSTGRES_USER" || -z "$POSTGRES_PASSWORD" ]]; then
  echo "Ошибка: Переменные окружения POSTGRES_DB, POSTGRES_USER или POSTGRES_PASSWORD не установлены."
  exit 1
fi

# Логируем значения переменных для отладки
echo "POSTGRES_DB: $POSTGRES_DB"
echo "POSTGRES_USER: $POSTGRES_USER"
echo "POSTGRES_PASSWORD: $POSTGRES_PASSWORD"

# Генерируем servers.json с подстановкой значений переменных
cat <<EOF > /pgadmin4/servers.json
{
    "Servers": {
        "1": {
            "Name": "PostgreSQL",
            "Group": "Servers",
            "Host": "postgres-db",
            "Port": 5432,
            "MaintenanceDB": "$POSTGRES_DB",
            "Username": "$POSTGRES_USER",
            "Password": "$POSTGRES_PASSWORD",
            "SSLMode": "prefer",
            "ConnectNow": true
        }
    }
}
EOF

# Запускаем pgAdmin
exec /entrypoint.sh "$@"
