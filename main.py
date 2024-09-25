from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QMainWindow, QApplication, QGridLayout, QFileDialog, QInputDialog
import pyqtgraph as pg
import sys
import os
import pandas as pd
from video_processor import VideoProcessor
from piv_processor import PIVProcessor
from resource_monitor import ResourceMonitor
from piv_logger import PIVLogger
from utils import create_new_session_folder, check_environment, ensure_folder_structure
import pyrealsense2 as rs
from queue import Queue
import threading
import numpy as np
import time
import cv2

class PIVApp(QMainWindow):
    def __init__(self):
        super(PIVApp, self).__init__()
        self.setWindowTitle("PIV Analyzer")
        self.setGeometry(0, 0, 1920, 1080)

        # Environment check: check Intel RealSense camera, CPU, GPU, Memory
        self.gpu_enabled = check_environment()

        # Initialize the camera
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        try:
            self.pipeline.start(config)
        except Exception as e:
            print(f"Camera not available: {e}")

        # Default save folder
        self.session_folder = create_new_session_folder()

        # Initialize processors and loggers
        self.video_processor = VideoProcessor(gpu_enabled=self.gpu_enabled)
        self.piv_processor = PIVProcessor(gpu_enabled=self.gpu_enabled)
        self.piv_logger = PIVLogger(self.session_folder)
        self.resource_monitor = ResourceMonitor()

        # Ensure folder structure is correct inside the session folder
        ensure_folder_structure(self.session_folder)

        # Queue for frame processing
        self.frame_queue = Queue()

        # Create UI components
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.pause_resume_button = QPushButton("Pause/Resume")
        self.record_button = QPushButton("Record")
        
        self.see_all_button = QPushButton("See All")
        self.swap_visualization_button = QPushButton("Swap Visualization")
        self.load_data_button = QPushButton("Load Data")
        self.save_folder_button = QPushButton("Set Save Folder")
        
        self.clean_video_display = QLabel("Clean Video Feed (Unprocessed)")
        self.piv_overlay_display = QLabel("PIV Overlay Video Feed (With PIV Quiver Plot)")
        self.velocity_plot_widget = pg.PlotWidget(title="Visualization Plot")

        self.cpu_usage_label = QLabel("CPU Usage: ")
        self.gpu_usage_label = QLabel("GPU Usage: ")
        self.memory_usage_label = QLabel("Memory Usage: ")

        # Set up grid layout
        grid_layout = QGridLayout()

        # Top row: video feeds (clean feed on left, PIV overlay on right)
        grid_layout.addWidget(self.clean_video_display, 0, 0)
        grid_layout.addWidget(self.piv_overlay_display, 0, 1)

        # Second row: control buttons (Start, Stop, Pause/Resume, Record)
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.pause_resume_button)
        control_layout.addWidget(self.record_button)
        grid_layout.addLayout(control_layout, 1, 0, 1, 2)  # Span across both columns

        # Third row: Visualization plot (real-time)
        grid_layout.addWidget(self.velocity_plot_widget, 2, 0, 1, 2)

        # Fourth row: Other buttons (See All, Swap Visualization, Load Data, Save Folder)
        options_layout = QHBoxLayout()
        options_layout.addWidget(self.see_all_button)
        options_layout.addWidget(self.swap_visualization_button)
        options_layout.addWidget(self.load_data_button)
        options_layout.addWidget(self.save_folder_button)
        grid_layout.addLayout(options_layout, 3, 0, 1, 2)

        # Fifth row: System resource usage (CPU, GPU, Memory)
        resource_layout = QHBoxLayout()
        resource_layout.addWidget(self.cpu_usage_label)
        resource_layout.addWidget(self.gpu_usage_label)
        resource_layout.addWidget(self.memory_usage_label)
        grid_layout.addLayout(resource_layout, 4, 0, 1, 2)

        # Set the central widget
        container = QWidget()
        container.setLayout(grid_layout)
        self.setCentralWidget(container)

        # Initialize frame skipping logic
        self.frame_skip_rate = 2  # Default: Process every 2nd frame
        self.current_frame = 0  # To track the current frame

        # ThreadPool for asynchronous processing
        self.executor = threading.Thread(target=self.collect_frames)

        # UI controls
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        self.pause_resume_button.clicked.connect(self.pause_resume_camera)
        self.record_button.clicked.connect(self.toggle_recording)
        self.see_all_button.clicked.connect(self.see_all_visualizations)
        self.swap_visualization_button.clicked.connect(self.swap_visualizations)
        self.load_data_button.clicked.connect(self.load_data)
        self.save_folder_button.clicked.connect(self.set_save_folder)

    def start_camera(self):
        self.is_running = True
        self.executor.start()

    def stop_camera(self):
        self.is_running = False
        self.pipeline.stop()
        self.video_processor.stop_video_recording()

    def pause_resume_camera(self):
        self.is_running = not self.is_running

    def toggle_recording(self):
        if self.is_running:
            self.stop_camera()
        else:
            self.start_camera()

    def see_all_visualizations(self):
        # Split the visualizations into four regions for real-time plotting
        self.magnitude_widget = pg.PlotWidget(title="Magnitude")
        self.streamlines_widget = pg.PlotWidget(title="Streamlines")
        self.vorticity_widget = pg.PlotWidget(title="Vorticity")
        self.energy_widget = pg.PlotWidget(title="Energy")

        # Set up a new grid layout for the four visualizations
        grid_layout = QGridLayout()

        # Add the four visualization widgets
        grid_layout.addWidget(self.magnitude_widget, 0, 0)
        grid_layout.addWidget(self.streamlines_widget, 0, 1)
        grid_layout.addWidget(self.vorticity_widget, 1, 0)
        grid_layout.addWidget(self.energy_widget, 1, 1)

        # Set the new layout
        container = QWidget()
        container.setLayout(grid_layout)
        self.setCentralWidget(container)

        # Update all visualizations in real-time
        self.update_all_visualizations()

    def update_all_visualizations(self):
        # Magnitude
        magnitude = np.sqrt(self.u**2 + self.v**2)
        self.magnitude_widget.clear()
        self.magnitude_widget.plot(magnitude.flatten())

        # Streamlines (placeholder for streamline drawing)
        self.streamlines_widget.clear()
        self.streamlines_widget.plot(np.sin(magnitude.flatten()))  # Example placeholder

        # Vorticity
        dx, dy = 1, 1  # Assuming grid spacing is uniform
        dvdx = np.gradient(self.v, dx, axis=1)
        dudy = np.gradient(self.u, dy, axis=0)
        vorticity = dvdx - dudy
        self.vorticity_widget.clear()
        self.vorticity_widget.plot(vorticity.flatten())

        # Energy (TKE)
        u_mean = np.mean(self.u)
        v_mean = np.mean(self.v)
        u_prime = self.u - u_mean
        v_prime = self.v - v_mean
        tke = 0.5 * (u_prime**2 + v_prime**2)
        self.energy_widget.clear()
        self.energy_widget.plot(tke.flatten())

    def swap_visualizations(self):
        # Allow user to choose between Magnitude, Streamlines, Vorticity, Energy
        options = ["Magnitude", "Streamlines", "Vorticity", "Energy"]
        chosen_option, ok = QInputDialog.getItem(self, "Choose Visualization", "Select visualization:", options, 0, False)
        
        if ok:
            # Clear the plot before displaying the new visualization
            self.velocity_plot_widget.clear()
            
            if chosen_option == "Magnitude":
                self.display_magnitude()
            elif chosen_option == "Streamlines":
                self.display_streamlines()
            elif chosen_option == "Vorticity":
                self.display_vorticity()
            elif chosen_option == "Energy":
                self.display_energy()

    def display_magnitude(self):
        # Calculate and display Magnitude of the velocity
        magnitude = np.sqrt(self.u**2 + self.v**2)
        self.velocity_plot_widget.clear()
        self.velocity_plot_widget.plot(magnitude.flatten())  # You may want to visualize this with a color plot

    def display_streamlines(self):
        # Plot streamlines using velocity field u, v
        plt.figure()
        plt.streamplot(self.x, self.y, self.u, self.v, density=2, color='b')
        plt.title('Streamlines')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    def display_vorticity(self):
        # Calculate and display Vorticity
        dx, dy = 1, 1  # Assuming grid spacing is uniform
        dvdx = np.gradient(self.v, dx, axis=1)
        dudy = np.gradient(self.u, dy, axis=0)
        vorticity = dvdx - dudy
        self.velocity_plot_widget.clear()
        self.velocity_plot_widget.plot(vorticity.flatten())

    def display_energy(self):
        # Calculate and display Turbulent Kinetic Energy (TKE)
        u_mean = np.mean(self.u)
        v_mean = np.mean(self.v)
        u_prime = self.u - u_mean
        v_prime = self.v - v_mean
        tke = 0.5 * (u_prime**2 + v_prime**2)
        self.velocity_plot_widget.clear()
        self.velocity_plot_widget.plot(tke.flatten())

    def load_data(self):
        # Load previously saved PIV data
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Data", "", "CSV Files (*.csv)")
        if file_name:
            try:
                df = pd.read_csv(file_name)
                # Display data in the corresponding widgets (can customize how to show)
                print("Loaded data:", df.head())
            except Exception as e:
                print(f"Error loading data: {e}")

    def set_save_folder(self):
        # Let user choose folder for saving data, otherwise default to session folder
        folder = QFileDialog.getExistingDirectory(self, "Select Save Folder")
        if folder:
            self.session_folder = folder
            # Ensure required subfolder structure exists
            ensure_folder_structure(self.session_folder)
        else:
            print("Using default session folder:", self.session_folder)

    def collect_frames(self):
        while self.is_running:
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if self.current_frame % self.frame_skip_rate == 0:
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # Submit frame processing to the thread pool
                self.executor = threading.Thread(target=self.process_and_visualize_frame, args=(depth_image, color_image,))
                self.executor.start()

            self.current_frame += 1

    def process_and_visualize_frame(self, depth_image, color_image):
        # Perform PIV processing
        piv_result = self.piv_processor.perform_piv(depth_image, depth_image)
        self.x, self.y, self.u, self.v = piv_result  # Save velocity components

        # Log the data
        timestamp = time.time()
        self.piv_logger.log_to_csv(timestamp, self.x, self.y, self.u, self.v, np.sqrt(self.u ** 2 + self.v ** 2), np.zeros_like(self.u), 0)
        self.piv_logger.log_to_json(timestamp, self.x, self.y, self.u, self.v)

        # Visualize in real-time
        self.display_real_time_quiver(self.x, self.y, self.u, self.v)

        # Overlay PIV on video and display
        piv_overlay_frame = self.piv_processor.overlay_piv_on_video(color_image, piv_result)
        self.display_piv_overlay(piv_overlay_frame)

    def display_real_time_quiver(self, x, y, u, v):
        # Plot quiver plot with real-time data
        self.velocity_plot_widget.clear()

        for i in range(len(x)):
            for j in range(len(y)):
                # Create vector based on velocity components u, v
                arrow = pg.ArrowItem(pos=(x[i, j], y[i, j]), angle=np.arctan2(v[i, j], u[i, j]) * 180 / np.pi, headLen=10)
                self.velocity_plot_widget.addItem(arrow)

        self.velocity_plot_widget.repaint()

    def display_piv_overlay(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        image = QtGui.QImage(frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(image)
        self.piv_overlay_display.setPixmap(pixmap)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PIVApp()
    window.show()
    sys.exit(app.exec_())
