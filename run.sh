#!/usr/bin/env bash
# ZentroMart VoiceOps Execution Script
set -e

if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

python run.py "$@"
