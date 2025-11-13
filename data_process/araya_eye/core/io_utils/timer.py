import time
from typing import Optional

class Timer:
    def __init__(self):
        self.elapsed_time:Optional[float] = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_time = time.perf_counter() - self.start_time

    def get(self)-> float:
        return self.elapsed_time if self.elapsed_time else 0.0