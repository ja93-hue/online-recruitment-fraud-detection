"""
Script to run both Flask backend and Streamlit frontend
"""
import subprocess
import sys
import time
import os
from pathlib import Path

# Change to app directory
os.chdir(Path(__file__).resolve().parent)


def run_backend():
    """Run Flask backend server."""
    print("Starting Flask backend server...")
    return subprocess.Popen(
        [sys.executable, "backend/api.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )


def run_frontend():
    """Run Streamlit frontend."""
    print("Starting Streamlit frontend...")
    return subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "frontend/app.py", 
         "--server.port=8501", "--server.address=localhost"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )


def main():
    print("="*60)
    print("Fake Job Detection Application")
    print("="*60)
    print()
    
    # Start backend
    backend_process = run_backend()
    print("Backend starting on http://localhost:5000")

    # Print backend output for debugging
    for i in range(10):  # Print first 10 lines (or until backend is ready)
        line = backend_process.stdout.readline()
        if not line:
            break
        print("[BACKEND]", line.decode(errors="replace").rstrip())
        if b"Running on" in line or b"Press CTRL+C" in line:
            break

    # Wait a bit more for backend to be ready
    time.sleep(2)

    # Start frontend
    frontend_process = run_frontend()
    print("Frontend starting on http://localhost:8501")

    print()
    print("="*60)
    print("Application is running!")
    print("- Backend API: http://localhost:5000")
    print("- Frontend UI: http://localhost:8501")
    print("="*60)
    print("Press Ctrl+C to stop both servers")

    try:
        # Keep running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        backend_process.terminate()
        frontend_process.terminate()
        print("Application stopped.")


if __name__ == "__main__":
    main()
