import time
import numpy as np


class Timer:

    def __init__(self):
        self.save_times = []
        self.start_time = 0

    def start(self):
        self.start_time = time.process_time()

    def end(self):
        end_time = time.process_time()
        self.save_times.append(end_time - self.start_time)

    def get_average_time(self):
        return np.mean(self.save_times)
