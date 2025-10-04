
# Project Name

This project provides an easy setup to run the application locally using shell scripts.

## Prerequisites

- Linux or macOS terminal (or WSL on Windows)
- `bash` and standard shell utilities
- Python 3.10+ installed
- Optional: `virtualenv` for a clean Python environment



## Setup and Run

### 1. Install Dependencies

Run the installation script to set up all required dependencies:

```bash
./install.sh
````

This script will:

* Install Python dependencies
* Set up environment variables if needed
* Prepare any required assets

---

### 2. Start the Application

Run the start script to launch the server:

```bash
./start.sh
```

Once the server starts, you should see a message like:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

---

### 3. Open in Browser

Open your web browser and navigate to:

[http://localhost:8000](http://localhost:8000)

You should see the application running.

---

### 4. Stop the Application

When you are done, run the stop script to cleanly shut down the server:

```bash
./stop.sh
```

---

## Notes

* Make sure `install.sh`, `start.sh`, and `stop.sh` are executable. You can make them executable using:

```bash
chmod +x install.sh start.sh stop.sh
```

* If you want to run the server on a different port, edit the `start.sh` script accordingly.



