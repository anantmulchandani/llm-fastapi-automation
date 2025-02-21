#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

IMAGE_NAME="llm_automation:v1"

echo "Building Docker image: $IMAGE_NAME"
docker build -t $IMAGE_NAME .

echo "Docker image '$IMAGE_NAME' built successfully!"
