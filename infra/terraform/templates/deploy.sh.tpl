#!/usr/bin/env bash
set -euo pipefail

AWS_REGION="${aws_region}"
ECR_URI="${ecr_uri}"
CONTAINER_NAME="${container_name}"
CONTAINER_PORT=${container_port}
HOST_PORT=${host_port}
IMAGE_TAG="${1:-latest}"

log() {
  printf "[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

ensure_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    log "Installing Docker..."
    sudo yum update -y
    sudo amazon-linux-extras install docker -y
    sudo systemctl enable --now docker
    sudo usermod -a -G docker ec2-user
  fi
}

login_ecr() {
  log "Logging into ECR..."
  aws ecr get-login-password --region "$AWS_REGION" \
    | docker login --username AWS --password-stdin "$ECR_URI"
}

pull_and_run() {
  log "Pulling $ECR_URI:$IMAGE_TAG"
  docker pull "$ECR_URI:$IMAGE_TAG"

  if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    log "Stopping existing container..."
    docker stop "$CONTAINER_NAME" || true
    docker rm "$CONTAINER_NAME" || true
  fi

  log "Starting container..."
  docker run -d \
    --name "$CONTAINER_NAME" \
    -p "$HOST_PORT:$CONTAINER_PORT" \
    --restart unless-stopped \
    "$ECR_URI:$IMAGE_TAG"

  log "Deployment complete."
}

ensure_docker
login_ecr
pull_and_run
