#!/usr/bin/env sh

set -e

# if ANALYTIC_PORT is set, use it; otherwise 5000
port=${ANALYTIC_PORT:-5000}
worker_timeout=${WORKER_TIMEOUT:-120}
# need a large timeout as things take a little bit to load
# echo "Starting up analytic with model file: $MODEL_PATH"
gunicorn -t $worker_timeout --bind 0.0.0.0:$port app_CE:app
