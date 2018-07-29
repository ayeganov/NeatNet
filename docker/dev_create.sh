#!/usr/bin/env bash

TIME=$(date  +%Y%m%d_%H%M)
if [ -z "${DOCKER_REPO}" ]; then
    DOCKER_REPO=ayeganov/neatnet
fi

ROOT_DIR="$( cd "$( dirname "$0" )/.." && pwd )"
ARCH=$(uname -m)
TAG="dev-${ARCH}-${TIME}"

echo "Root dir: ${ROOT_DIR}"

# Build image from ROOT_DIR, while use the specified Dockerfile.
docker build -t "${DOCKER_REPO}:${TAG}" \
    -f "${ROOT_DIR}/docker/dockerfile" \
    "${ROOT_DIR}"
