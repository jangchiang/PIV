import numpy as np
import openpiv.process
import openpiv.scaling
import cv2

class PIVProcessor:
    def __init__(self, scale_factor=10, gpu_enabled=True):
        self.scale_factor = scale_factor
        self.gpu_enabled = gpu_enabled

    def perform_piv(self, frame1, frame2):
        # Use CuPy for GPU-based calculations if enabled
        if self.gpu_enabled:
            frame1 = cp.asarray(frame1)
            frame2 = cp.asarray(frame2)

            # Perform GPU-accelerated PIV
            u, v, sig2noise = openpiv.process.extended_search_area_piv(
                frame1, frame2, window_size=24, overlap=12, dt=0.02,
                search_area_size=48, sig2noise_method='peak2peak'
            )
            u, v = cp.asnumpy(u), cp.asnumpy(v)
        else:
            u, v, sig2noise = openpiv.process.extended_search_area_piv(
                frame1, frame2, window_size=24, overlap=12, dt=0.02,
                search_area_size=48, sig2noise_method='peak2peak'
            )

        x, y, u, v = openpiv.scaling.uniform(u, v, scaling_factor=1)
        return x, y, u, v

    def overlay_piv_on_video(self, video_frame, piv_result):
        x, y, u, v = piv_result
        for i in range(len(x)):
            for j in range(len(y)):
                start_point = (int(x[i, j]), int(y[i, j]))
                end_point = (int(x[i, j] + u[i, j] * self.scale_factor), int(y[i, j] + v[i, j] * self.scale_factor))
                cv2.arrowedLine(video_frame, start_point, end_point, (0, 255, 0), 1, tipLength=0.3)
        return video_frame
