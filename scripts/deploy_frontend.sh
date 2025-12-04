#!/usr/bin/env bash
set -euo pipefail

# Frontend auto-deployment helper
# Builds the Vite React app, syncs the output to the Nginx web root, and reloads Nginx.
#
# Usage:
#   ./deploy_frontend.sh
#
# Optional environment overrides:
#   FRONTEND_DIR      - path to the frontend source (default: ../frontend)
#   TARGET_DIR        - Nginx document root to receive the build (default: /var/www/inaivers)
#   NGINX_SERVICE     - systemd service name for Nginx (default: nginx)
#   NPM_COMMAND       - npm executable to use (default: npm)
#   SKIP_NPM_INSTALL  - set to 1 to skip `npm install` (assumes node_modules are ready)
#   SKIP_BUILD        - set to 1 to skip `npm run build` (assumes dist already present)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FRONTEND_DIR="${FRONTEND_DIR:-${REPO_ROOT}/frontend}"
TARGET_DIR="${TARGET_DIR:-/var/www/inaivers}"
NGINX_SERVICE="${NGINX_SERVICE:-nginx}"
NPM_COMMAND="${NPM_COMMAND:-npm}"
BUILD_DIR="${BUILD_DIR:-${FRONTEND_DIR}/dist}"
SKIP_NPM_INSTALL_RAW="${SKIP_NPM_INSTALL:-0}"
SKIP_BUILD_RAW="${SKIP_BUILD:-0}"

normalize_bool() {
  case "$1" in
    1|true|TRUE|yes|YES|on|ON) echo 1 ;;
    *) echo 0 ;;
  esac
}

SKIP_NPM_INSTALL="$(normalize_bool "$SKIP_NPM_INSTALL_RAW")"
SKIP_BUILD="$(normalize_bool "$SKIP_BUILD_RAW")"

if [[ $EUID -ne 0 ]]; then
  SUDO="sudo"
else
  SUDO=""
fi

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log "ERROR: '$1' not found in PATH."
    exit 1
  fi
}

if [[ "$SKIP_NPM_INSTALL" != "1" || "$SKIP_BUILD" != "1" ]]; then
  require_cmd "$NPM_COMMAND"
fi
require_cmd rsync
require_cmd systemctl

log "Using frontend directory: ${FRONTEND_DIR}"
log "Target deployment directory: ${TARGET_DIR}"
log "Nginx service: ${NGINX_SERVICE}"
log "Flags -> SKIP_NPM_INSTALL=${SKIP_NPM_INSTALL} (raw=${SKIP_NPM_INSTALL_RAW}), SKIP_BUILD=${SKIP_BUILD} (raw=${SKIP_BUILD_RAW})"

if [[ ! -d "$FRONTEND_DIR" ]]; then
  log "ERROR: Frontend directory '${FRONTEND_DIR}' does not exist."
  exit 1
fi

if [[ "$SKIP_NPM_INSTALL" == "1" ]]; then
  log "Skipping npm install (SKIP_NPM_INSTALL=1)"
else
  log "Installing frontend dependencies (if needed)"
  (cd "$FRONTEND_DIR" && "$NPM_COMMAND" install --legacy-peer-deps >/dev/null)
fi

if [[ "$SKIP_BUILD" == "1" ]]; then
  log "Skipping npm run build (SKIP_BUILD=1)"
else
  log "Building production bundle"
  (cd "$FRONTEND_DIR" && "$NPM_COMMAND" run build)
fi

if [[ ! -d "$BUILD_DIR" ]]; then
  log "ERROR: Build output not found at '${BUILD_DIR}'."
  exit 1
fi

log "Syncing build to ${TARGET_DIR}"
$SUDO mkdir -p "$TARGET_DIR"
$SUDO rsync -av --delete "${BUILD_DIR}/" "$TARGET_DIR/"

log "Reloading Nginx"
$SUDO systemctl reload "$NGINX_SERVICE"

log "Deployment completed successfully. Frontend is now live at https://inaivers.com/"
