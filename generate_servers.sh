#!/bin/bash

echo "====> Starting generate_servers.sh"

# Проверяем, что переменные окружения установлены
if [[ -z "$POSTGRES_DB" || -z "$POSTGRES_USER" || -z "$POSTGRES_PASSWORD" ]]; then
  echo "Ошибка: Не установлены переменные окружения POSTGRES_DB, POSTGRES_USER или POSTGRES_PASSWORD."
  exit 1
fi

# Логируем переменные
echo "POSTGRES_DB: $POSTGRES_DB"
echo "POSTGRES_USER: $POSTGRES_USER"
echo "POSTGRES_PASSWORD: $POSTGRES_PASSWORD"

# Исправляем права доступа
echo "====> Fixing permissions for /pgadmin4"
chmod -R 777 /pgadmin4 || { echo "Ошибка: Не удалось изменить права доступа на /pgadmin4"; exit 1; }

# Генерируем servers.json
echo "====> Generating servers.json"
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

# Логируем содержимое файла
echo "====> Generated servers.json:"
cat /pgadmin4/servers.json

# Запускаем pgAdmin
echo "====> Starting pgAdmin"
exec /entrypoint.sh "$@"
