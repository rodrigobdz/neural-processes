#!/usr/bin/env bash
#
# Shared functions and variables across scripts.

set -o errexit
set -o pipefail
set -o nounset

readonly LOG_DIR='./log'
# Automatically create log dir, if it doesn't exist already
mkdir -p "$LOG_DIR"

timestamp() {
  # Default timezone to date's built-in format
  # macOS outputs code e.g. CET whereas Linux as time difference +08
  local timezone=%Z
  # Set timezone to contents of system file if exists
  if [ -f /etc/timezone ]; then
    timezone=$(cat /etc/timezone)
  fi
  date "+%a %b %d %I:%M:%S %p $timezone %Y"
}

fancy_echo() {
  echo
  timestamp
  echo "[NEURAL PROCESSES] ==> $1"
  echo
}

err_exit() {
  echo
  echo "ERROR: $1"
  echo
  exit 1
}

summary() {
  fancy_echo "Logs written to $LOG_FILENAME"
  fancy_echo "Done"
}
