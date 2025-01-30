#!/bin/bash

cat <<EOF > /pgadmin4/servers.json
{
    "Servers": {
        "1": {
            "Name": "PostgreSQL",
            "Group": "Servers",
            "Host": "postgres-db",
            "Port": 5432,
            "MaintenanceDB": "${POSTGRES_DB}",
            "Username": "${POSTGRES_USER}",
            "Password": "${POSTGRES_PASSWORD}",
            "SSLMode": "prefer",
            "ConnectNow": true
        }
    }
}
EOF

exec /entrypoint.sh
