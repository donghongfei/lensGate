#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="lensgate"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_USER="$(whoami)"
NODE_BIN="$(command -v node)"
ENV_FILE="${SCRIPT_DIR}/.env"

echo "Installing ${SERVICE_NAME} systemd service..."
echo "  WorkingDirectory : ${SCRIPT_DIR}"
echo "  User             : ${CURRENT_USER}"
echo "  Node             : ${NODE_BIN}"
if [ -f "${ENV_FILE}" ]; then
  echo "  EnvironmentFile  : ${ENV_FILE}"
else
  echo "  EnvironmentFile  : (not found, using defaults)"
fi

# EnvironmentFile 行：只有 .env 存在时才写入
ENV_FILE_LINE=""
if [ -f "${ENV_FILE}" ]; then
  ENV_FILE_LINE="EnvironmentFile=${ENV_FILE}"
fi

sudo tee "${SERVICE_FILE}" > /dev/null <<EOF
[Unit]
Description=LensGate OpenAI Proxy
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${CURRENT_USER}
WorkingDirectory=${SCRIPT_DIR}
${ENV_FILE_LINE}
ExecStart=${NODE_BIN} --env-file-if-exists=.env dist/server.js
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"
sudo systemctl restart "${SERVICE_NAME}"

echo ""
echo "Done. Useful commands:"
echo "  sudo systemctl status ${SERVICE_NAME}"
echo "  sudo journalctl -u ${SERVICE_NAME} -f"
echo "  sudo systemctl restart ${SERVICE_NAME}"
