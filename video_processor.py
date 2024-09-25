import cv2

class VideoProcessor:
    def __init__(self, gpu_enabled=True):
        self.gpu_enabled = gpu_enabled
        self.clean_feed_writer = None
        self.piv_overlay_writer = None

    def start_video_recording(self, clean_feed_path, piv_overlay_path):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.clean_feed_writer = cv2.VideoWriter(clean_feed_path, fourcc, 30.0, (640, 480))
        self.piv_overlay_writer = cv2.VideoWriter(piv_overlay_path, fourcc, 30.0, (640, 480))

    def stop_video_recording(self):
        if self.clean_feed_writer is not None:
            self.clean_feed_writer.release()
            self.clean_feed_writer = None
        if self.piv_overlay_writer is not None:
            self.piv_overlay_writer.release()
            self.piv_overlay_writer = None
