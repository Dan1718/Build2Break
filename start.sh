#!/bin/bash

# --- Configuration (should match your setup script) ---
VENV_NAME="my_project_venv"
CONDA_ENV_NAME="data_analysis_py312"
PID_DIR="pids"

# --- Create a directory to store process IDs ---
mkdir -p $PID_DIR

# --- Helper function for Conda activation ---
# We need to source the conda script to make the 'conda' command available
# This path should match where you installed Miniconda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "🚀 Starting all services..."
echo "--------------------------------"

# --- Start Frontend Service in the venv ---
echo "[1/3] Starting Frontend service on port 8001..."
(
  source "$VENV_NAME/bin/activate"
  uvicorn Frontend.main:app --port 8001 &
  echo $! > "$PID_DIR/frontend.pid"
)
sleep 2 # Give it a moment to start up

# --- Start Text Service in the Conda env ---
echo "[2/3] Starting Text service on port 8002..."
(
  conda activate "$CONDA_ENV_NAME"
  uvicorn text.main:app --port 8002 &
  echo $! > "$PID_DIR/text.pid"
)
sleep 2 # Give it a moment to start up

# --- Start Video Service in the Conda env ---
echo "[3/3] Starting Video service on port 8003..."
(
  conda activate "$CONDA_ENV_NAME"
  uvicorn video.main:app --port 8003 &
  echo $! > "$PID_DIR/video.pid"
)

echo "--------------------------------"
echo "✅ All services have been launched in the background."
echo "Process IDs are stored in the '$PID_DIR/' directory."
echo "Run './stop_services.sh' to terminate them."