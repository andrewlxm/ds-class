#!/bin/bash
set -e

IMAGE="data-science-app"
CONTAINER="data-science-app"

docker build -t "$IMAGE" .

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    docker rm -f "$CONTAINER"
fi

docker run --privileged --name "$CONTAINER" -v "$(pwd):/app" "$IMAGE" \
    bash -c "echo 1 > /proc/sys/kernel/perf_event_paranoid && python estimate.py"
