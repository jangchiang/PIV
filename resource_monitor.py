import psutil
import GPUtil

class ResourceMonitor:
    def __init__(self):
        pass

    def monitor_resources(self):
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory().percent
        gpu_usage = None
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_usage = gpus[0].load * 100
        except:
            pass
        return cpu_usage, memory, gpu_usage
