import os
import cv2
import numpy as np
import openpiv.process
import openpiv.scaling
from PyQt5 import QtWidgets, QtGui, uic
from PyQt5.QtWidgets import QFileDialog
from datetime import datetime
import pyrealsense2 as rs
import math
import matplotlib.pyplot as plt
import json
import cupy as cp  # For GPU acceleration
import GPUtil  # For GPU utilization monitoring
import psutil  # For CPU and memory utilization monitoring
from concurrent.futures import ThreadPoolExecutor
from memory_profiler import profile
import csv
import time
from queue import Queue
import threading
import pyqtgraph as pg  # For real-time charting
import pandas as pd  # For loading CSV

# Global ThreadPoolExecutor for multiprocessing
executor = ThreadPoolExecutor()

# Set the interval for auto-saving the visualizations (1 minute = 60 seconds)
AUTO_SAVE_INTERVAL = 60

# Folder paths
folders = {
    'clean_feed': 'CleanFeed',
    'piv_video': 'PIVVideo',
    'csv': 'CSVData',
    'magnitude': 'Magnitude',
    'streamlines': 'Streamlines',
    'vorticity': 'Vorticity',
    'energy': 'Energy',
    'clean_feed_video': 'CleanFeedVideo',  # New folder for Clean Feed Video
    'piv_overlay_video': 'PIVOverlayVideo'  # New folder for PIV Overlay Video
}

class VideoProcessor:
    def __init__(self, gpu_enabled=True):
        self.gpu_enabled = gpu_enabled
        self.clean_feed_writer = None
        self.piv_overlay_writer = None

    def start_video_recording(self, clean_feed_path, piv_overlay_path):
        """Start recording video for clean feed and PIV overlay."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 format
        
        # Initialize the VideoWriter for clean feed
        self.clean_feed_writer = cv2.VideoWriter(clean_feed_path, fourcc, 30.0, (640, 480))
        
        # Initialize the VideoWriter for PIV overlay
        self.piv_overlay_writer = cv2.VideoWriter(piv_overlay_path, fourcc, 30.0, (640, 480))

    def stop_video_recording(self):
        """Stop recording video and release the VideoWriters."""
        if self.clean_feed_writer is not None:
            self.clean_feed_writer.release()
            self.clean_feed_writer = None
        if self.piv_overlay_writer is not None:
            self.piv_overlay_writer.release()
            self.piv_overlay_writer = None

    def process_frame(self, color_frame):
        if self.gpu_enabled:
            # Upload frame to GPU
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(color_frame)

            # Perform GPU-based operations (example: resizing)
            resized_gpu_frame = cv2.cuda.resize(gpu_frame, (640, 480))

            # Download back to CPU for further processing
            processed_frame = resized_gpu_frame.download()
        else:
            # Fallback to CPU-based processing
            processed_frame = cv2.resize(color_frame, (640, 480))

        return processed_frame

class PIVApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(PIVApp, self).__init__()
        uic.loadUi('piv_gui.ui', self)

        # Set the window title to "PIV Analyzer"
        self.setWindowTitle("PIV Analyzer")

        # Set the window to open in full-screen mode but allow minimizing
        self.setGeometry(0, 0, 1920, 1080)  # Set the initial size to fit a 1920x1080 resolution
        self.showMaximized()  # Start the application in full-screen mode

        self.pipeline = rs.pipeline()

        # Configuration for the RealSense camera
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        try:
            self.pipeline.start(config)
        except Exception as e:
            print(f"Camera not available: {e}")
            self.show_system_check()

        # Create session folder and folders for all visualizations
        self.session_folder = self.create_new_session_folder()

        for folder in folders.values():
            folder_path = os.path.join(self.session_folder, folder)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

        self.video_processor = VideoProcessor(gpu_enabled=True)
        self.is_running = False
        self.is_paused = False
        self.frame_queue = Queue()

        # Real-time plotting for velocity magnitude
        self.plot_widget = pg.PlotWidget(self)
        self.plot_widget.setGeometry(20, 400, 400, 300)
        self.plot_data = []

        # Set up resource monitor labels
        self.cpu_label = QtWidgets.QLabel(self)
        self.cpu_label.setGeometry(450, 10, 300, 30)
        self.cpu_label.setText("CPU Usage: N/A")

        self.gpu_label = QtWidgets.QLabel(self)
        self.gpu_label.setGeometry(450, 40, 300, 30)
        self.gpu_label.setText("GPU Usage: N/A")

        self.memory_label = QtWidgets.QLabel(self)
        self.memory_label.setGeometry(450, 70, 300, 30)
        self.memory_label.setText("Memory Usage: N/A")

        self.resource_monitor = ResourceMonitor(self.cpu_label, self.gpu_label, self.memory_label)

        # UI controls
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        self.record_button.clicked.connect(self.toggle_recording)

    def create_new_session_folder(self):
        base_dir = 'PIV_Sessions'
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        session_folder = os.path.join(base_dir, f'session_{timestamp}')
        os.makedirs(session_folder)
        return session_folder

    def start_camera(self):
        self.is_running = True
        frame_count = 0

        # Start video recording for clean feed and PIV overlay
        clean_feed_video_path = os.path.join(self.session_folder, folders['clean_feed_video'], 'clean_feed.mp4')
        piv_overlay_video_path = os.path.join(self.session_folder, folders['piv_overlay_video'], 'piv_overlay.mp4')
        self.video_processor.start_video_recording(clean_feed_video_path, piv_overlay_video_path)

        # Start a thread for frame collection
        frame_collector_thread = threading.Thread(target=self.collect_frames)
        frame_collector_thread.start()

        # Main loop for frame processing
        while self.is_running:
            if not self.is_paused and not self.frame_queue.empty():
                depth_frame, color_frame = self.frame_queue.get()

                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                piv_overlay_frame = self.process_piv(depth_image, color_image)

                # Display clean video and PIV overlay
                self.display_clean_video(color_image)
                self.display_piv_overlay(piv_overlay_frame)

                # Save clean feed and PIV overlay to video
                if self.video_processor.clean_feed_writer is not None:
                    self.video_processor.clean_feed_writer.write(color_image)

                if self.video_processor.piv_overlay_writer is not None:
                    self.video_processor.piv_overlay_writer.write(piv_overlay_frame)

    def stop_camera(self):
        self.is_running = False
        self.pipeline.stop()
        self.video_processor.stop_video_recording()

    def toggle_recording(self):
        # This method now controls the video saving process
        if self.is_running:
            self.stop_camera()
        else:
            self.start_camera()

    def display_clean_video(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        image = QtGui.QImage(frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(image)
        self.clean_video_display.setPixmap(pixmap)

    def display_piv_overlay(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        image = QtGui.QImage(frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(image)
        self.piv_overlay_display.setPixmap(pixmap)

    def collect_frames(self):
        while self.is_running:
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            self.frame_queue.put((depth_frame, color_frame))

    def process_piv(self, depth_image, color_image):
        piv_result = self.piv_processor.perform_piv(depth_image, depth_image)
        piv_overlay_frame = self.piv_processor.overlay_piv_on_video(color_image, piv_result)
        return piv_overlay_frame

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = PIVApp()
    window.show()
    app.exec_()
