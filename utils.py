import os
import GPUtil
import psutil
from datetime import datetime

import os
from datetime import datetime

def create_new_session_folder():
    """Creates a new session folder based on the timestamp."""
    base_dir = 'PIV_Sessions'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    session_folder = os.path.join(base_dir, f'session_{timestamp}')
    os.makedirs(session_folder)
    ensure_folder_structure(session_folder)  # Ensure the required subfolders exist
    return session_folder

def ensure_folder_structure(base_folder):
    """Ensure the required folder structure exists in the base folder."""
    folders = [
        "CleanFeed", "PIVVideo", "CSVData", "Magnitude", "Streamlines",
        "Vorticity", "Energy", "CleanFeedVideo", "PIVOverlayVideo"
    ]
    
    for folder in folders:
        folder_path = os.path.join(base_folder, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created missing folder: {folder_path}")

def check_environment():
    """Check hardware environment for Intel RealSense, CPU, GPU, and memory."""
    # Check CPU usage
    cpu_usage = psutil.cpu_percent()
    if cpu_usage > 80:
        print("Warning: High CPU usage detected.")

    # Check available memory
    memory_info = psutil.virtual_memory()
    if memory_info.available / (1024 ** 3) < 2:  # less than 2 GB available
        print("Warning: Low available memory.")

    # Check for GPU
    try:
        gpus = GPUtil.getGPUs()
        if gpus and gpus[0].load * 100 < 80:  # Ensure the GPU is available and not fully loaded
            print(f"GPU found: {gpus[0].name}")
            return True  # GPU is available
        else:
            print("GPU not found or under high load. Switching to CPU.")
            return False  # No GPU or it's under load
    except Exception as e:
        print("No GPU detected. Using CPU.")
        return False  # Fallback to CPU if no GPU
