#!/bin/bash

echo "====> Starting generate_servers.sh"

# Убедитесь, что переменные окружения установлены
if [[ -z "$POSTGRES_DB" || -z "$POSTGRES_USER" || -z "$POSTGRES_PASSWORD" ]]; then
  echo "Ошибка: Переменные окружения POSTGRES_DB, POSTGRES_USER или POSTGRES_PASSWORD не установлены."
  exit 1
fi

# Логируем переменные
echo "POSTGRES_DB: $POSTGRES_DB"
echo "POSTGRES_USER: $POSTGRES_USER"
echo "POSTGRES_PASSWORD: $POSTGRES_PASSWORD"

# Путь для сохранения файла servers.json
SERVERS_JSON_PATH="/var/lib/pgadmin/servers.json"

# Генерация файла servers.json
echo "====> Generating servers.json at $SERVERS_JSON_PATH"
cat <<EOF > "$SERVERS_JSON_PATH"
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

if [[ $? -ne 0 ]]; then
  echo "Ошибка: Не удалось создать $SERVERS_JSON_PATH"
  exit 1
fi

# Настройка прав доступа к файлу
echo "====> Setting permissions for servers.json"
chmod 777 "$SERVERS_JSON_PATH"

if [[ $? -ne 0 ]]; then
  echo "Ошибка: Не удалось настроить права для $SERVERS_JSON_PATH"
  exit 1
fi

# Проверяем содержимое файла
echo "====> Generated servers.json:"
cat "$SERVERS_JSON_PATH"

# Запускаем pgAdmin
echo "====> Starting pgAdmin"
exec /entrypoint.sh "$@"
