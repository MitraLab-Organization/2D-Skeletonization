#!/bin/bash
# Wrapper script to run batch_process.py with the virtual environment

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

./venv/bin/python batch_process.py "$@"
