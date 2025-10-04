#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
VENV_NAME="my_project_venv"
CONDA_ENV_NAME="data_analysis_py312"
PYTHON_VERSION_CONDA="3.12"
MINICONDA_INSTALL_PATH="$HOME/miniconda3"

# --- Helper Functions for Colorized Output ---
print_info() {
    echo -e "\n\e[1;34m[INFO] $1\e[0m"
}

print_success() {
    echo -e "\e[1;32m[SUCCESS] $1\e[0m"
}

print_warning() {
    echo -e "\e[1;33m[WARNING] $1\e[0m"
}


# ==============================================================================
# STEP 1: INSTALL SYSTEM DEPENDENCIES (PYTHON & BUILD TOOLS)
# ==============================================================================
print_info "Checking for system dependencies like Python..."

# This part is distribution-dependent. The user should uncomment the relevant section.
# For Debian/Ubuntu:
# sudo apt-get update && sudo apt-get install -y python3 python3-pip python3-venv build-essential wget
#
# For RedHat/CentOS/Fedora:
# sudo dnf groupinstall -y "Development Tools" && sudo dnf install -y python3 python3-pip wget
#
# For macOS (assuming Homebrew is installed):
# brew install python wget

# A simple check to see if python3 is available
if ! command -v python3 &> /dev/null
then
    print_warning "Python 3 is not found. Please install it using your system's package manager."
    exit 1
else
    print_success "Python 3 found."
fi


# ==============================================================================
# STEP 2: INSTALL MINICONDA
# ==============================================================================
print_info "Installing Miniconda..."

if [ -d "$MINICONDA_INSTALL_PATH" ]; then
    print_warning "Miniconda seems to be already installed at $MINICONDA_INSTALL_PATH. Skipping installation."
else
    # Download and install the latest Miniconda for Linux 64-bit
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda_installer.sh
    bash miniconda_installer.sh -b -p $MINICONDA_INSTALL_PATH
    rm miniconda_installer.sh
    print_success "Miniconda installed successfully to $MINICONDA_INSTALL_PATH"

    # Initialize Conda for bash shell
    print_info "Initializing Conda for the shell..."
    "$MINICONDA_INSTALL_PATH/bin/conda" init bash
    print_warning "Conda has been initialized. Please run 'source ~/.bashrc' or open a new terminal for changes to take effect before proceeding with Conda commands manually."
fi

# Add conda to the current session's PATH
export PATH="$MINICONDA_INSTALL_PATH/bin:$PATH"


# ==============================================================================
# STEP 3: CREATE AND SET UP A STANDARD PYTHON VIRTUAL ENVIRONMENT (venv)
# ==============================================================================
print_info "Creating a standard Python virtual environment named '$VENV_NAME'..."

if [ -d "$VENV_NAME" ]; then
    print_warning "Virtual environment '$VENV_NAME' already exists. Skipping creation."
else
    python3 -m venv $VENV_NAME
    print_success "Virtual environment '$VENV_NAME' created."
fi

print_info "Activating '$VENV_NAME' and installing packages from requirements-1.txt..."

# Activate the venv and install packages
source $VENV_NAME/bin/activate
pip install -r requirements-1.txt

# Deactivate after installation
deactivate
print_success "Packages installed in '$VENV_NAME'. Environment deactivated."


# ==============================================================================
# STEP 4: CREATE AND SET UP A CONDA ENVIRONMENT
# ==============================================================================
print_info "Creating a Conda environment named '$CONDA_ENV_NAME' with Python $PYTHON_VERSION_CONDA..."

# Check if the conda environment already exists
if conda env list | grep -q "$CONDA_ENV_NAME"; then
    print_warning "Conda environment '$CONDA_ENV_NAME' already exists. Skipping creation."
else
    conda create --name $CONDA_ENV_NAME python=$PYTHON_VERSION_CONDA -y
    print_success "Conda environment '$CONDA_ENV_NAME' created."
fi

print_info "Installing packages into '$CONDA_ENV_NAME' from req-2.txt..."

# Install packages into the specified conda environment without activating it
# The '-n' flag targets the environment.
conda install -n $CONDA_ENV_NAME --file req-2.txt -y
print_success "Packages installed in '$CONDA_ENV_NAME'."


# ==============================================================================
# FINAL INSTRUCTIONS
# ==============================================================================
print_info "----------------- SETUP COMPLETE -----------------"
print_success "All environments and packages have been set up."
echo ""
echo "To use the standard virtual environment, run:"
echo "  source $VENV_NAME/bin/activate"
echo ""
echo "To use the Conda environment, run:"
echo "  conda activate $CONDA_ENV_NAME"
echo ""
print_warning "If the 'conda' command is not found, you may need to open a new terminal or run: source ~/.bashrc"