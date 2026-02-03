#!/bin/bash
# build.sh - Docker + uv workflow for Neuron docs

set -e
IMAGE_NAME="neuron-docs"

case "${1:-build}" in
  build)
    docker build -t "$IMAGE_NAME" .
    ;;
  html)
    docker run --rm -v "$(pwd):/docs" "$IMAGE_NAME" -c "sphinx-build -b html . _build/html"
    ;;
  shell)
    docker run --rm -it -v "$(pwd):/docs" "$IMAGE_NAME"
    ;;
  clean)
    rm -rf _build
    ;;
  *)
    echo "Usage: $0 {build|html|shell|clean}"
    exit 1
    ;;
esac
