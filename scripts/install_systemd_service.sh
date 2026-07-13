#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="litellm-pgvector"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEMPLATE_FILE="${APP_DIR}/systemd/${SERVICE_NAME}.service"
SYSTEMD_UNIT_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
ENV_FILE="/etc/default/${SERVICE_NAME}"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run as root: sudo $0"
  exit 1
fi

RUN_USER="${SUDO_USER:-${USER:-root}}"
RUN_GROUP="$(id -gn "${RUN_USER}")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required but was not found"
  exit 1
fi

if [[ ! -f "${TEMPLATE_FILE}" ]]; then
  echo "Missing systemd template: ${TEMPLATE_FILE}"
  exit 1
fi

if [[ ! -d "${APP_DIR}" ]]; then
  echo "App directory not found: ${APP_DIR}"
  exit 1
fi

if [[ ! -f "${APP_DIR}/requirements.txt" ]]; then
  echo "requirements.txt not found in ${APP_DIR}"
  exit 1
fi

echo "Preparing Python virtual environment in ${APP_DIR}/.venv"
python3 -m venv "${APP_DIR}/.venv"
"${APP_DIR}/.venv/bin/pip" install --upgrade pip
"${APP_DIR}/.venv/bin/pip" install -r "${APP_DIR}/requirements.txt"

if [[ -f "${APP_DIR}/prisma/schema.prisma" ]]; then
  echo "Generating Prisma client"
  if ! (
    cd "${APP_DIR}"
    "${APP_DIR}/.venv/bin/prisma" generate
  ); then
    echo "Warning: Prisma client generation failed; continuing with existing client if present"
  fi
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  cat > "${ENV_FILE}" <<'EOF'
# Service runtime configuration for litellm-pgvector
DATABASE_URL=postgresql://postgres:postgres@localhost:15432/pgvector_store?schema=public
SERVER_API_KEY=change-me
UI_USERNAME=admin
UI_PASSWORD=change-me-now
EMBEDDING__MODEL=text-embedding-ada-002
EMBEDDING__BASE_URL=http://localhost:4000
EMBEDDING__API_KEY=sk-1234
EMBEDDING__DIMENSIONS=1536
HOST=0.0.0.0
PORT=18001
EOF
  echo "Created ${ENV_FILE}. Update secrets and endpoints before production use."
else
  echo "Using existing ${ENV_FILE}"
fi

sed \
  -e "s|{{APP_DIR}}|${APP_DIR}|g" \
  -e "s|{{RUN_USER}}|${RUN_USER}|g" \
  -e "s|{{RUN_GROUP}}|${RUN_GROUP}|g" \
  "${TEMPLATE_FILE}" > "${SYSTEMD_UNIT_FILE}"

chmod 0644 "${SYSTEMD_UNIT_FILE}"

systemctl daemon-reload
systemctl enable "${SERVICE_NAME}"
systemctl restart "${SERVICE_NAME}"

systemctl --no-pager --full status "${SERVICE_NAME}" || true

echo
echo "Service installed and enabled: ${SERVICE_NAME}"
echo "Configuration file: ${ENV_FILE}"
echo "Logs: journalctl -u ${SERVICE_NAME} -f"
