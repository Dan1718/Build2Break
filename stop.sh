#!/bin/bash

PID_DIR="pids"

echo "🛑 Stopping all services..."
echo "--------------------------------"

if [ ! -d "$PID_DIR" ]; then
  echo "PID directory not found. No services to stop."
  exit 0
fi

for pid_file in $PID_DIR/*.pid; do
  if [ -f "$pid_file" ]; then
    PID=$(cat "$pid_file")
    SERVICE_NAME=$(basename "$pid_file" .pid)

    # Check if the process is still running
    if ps -p $PID > /dev/null; then
      echo "Killing $SERVICE_NAME process with PID $PID..."
      kill $PID
      rm "$pid_file"
    else
      echo "Process for $SERVICE_NAME (PID $PID) is not running. Cleaning up PID file."
      rm "$pid_file"
    fi
  fi
done

echo "--------------------------------"
echo "✅ All services stopped and PID files cleaned up."