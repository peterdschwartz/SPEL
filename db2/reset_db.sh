#!/bin/bash
set -euo pipefail

### >>> EDIT THESE 4-6 LINES TO MATCH YOUR SETUP <<<
DB_NAME="spelapp"
DB_USER="djuser"
DB_PASS="197326"
DB_HOST="127.0.0.1"         # usually 127.0.0.1
DB_PORT="8000"

# Paths for your Django project on this machine
### <<< EDIT ABOVE >>>

echo "About to DROP and RECREATE database '$DB_NAME' and user '$DB_USER' on $DB_HOST:$DB_PORT."
read -rp "This will DESTROY any data in '$DB_NAME'. Continue? (yes/NO) " ans
[[ "${ans:-}" == "yes" ]] || { echo "Aborted."; exit 1; }

echo "==> Resetting MariaDB schema and user..."
# Use socket auth via sudo (Ubuntu default). If you use a root password, {

# Sanity checks
mariadb -u root -p <<SQL

DROP DATABASE IF EXISTS \`$DB_NAME\`;
CREATE DATABASE \`$DB_NAME\`
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

-- Create/replace user (localhost-only)
DROP USER IF EXISTS '$DB_USER'@'localhost';
CREATE USER '$DB_USER'@'localhost' IDENTIFIED BY '$DB_PASS';
GRANT ALL PRIVILEGES ON \`$DB_NAME\`.* TO '$DB_USER'@'localhost';
FLUSH PRIVILEGES;
SQL

echo "==> Done."
echo "Database: $DB_NAME        User: $DB_USER"

python3 manage.py makemigrations
python3 manage.py migrate
python3 manage.py update_all_data --all
python3 manage.py import_presets
