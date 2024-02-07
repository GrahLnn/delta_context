import requests
import time
import statistics
import threading
from collections import deque


class NetworkLatencyChecker:
    MAX_SAMPLES = 10

    def __init__(self, url="http://www.google.com/generate_204"):
        self.url = url
        self.delays = deque(maxlen=self.MAX_SAMPLES)
        self.thread = threading.Thread(target=self._measure_latency)
        self.thread.daemon = True

    def start(self):
        """启动延迟检测"""
        self.thread.start()

        waiting_time = 10
        while waiting_time > 0:
            print(
                f"Waiting {waiting_time} seconds for network latency check...",
                end="\r",
                flush=True,
            )
            time.sleep(1)
            waiting_time -= 1
        print()

    def _measure_latency(self):
        # count = 0
        while True:
            try:
                response = requests.get(self.url, timeout=5)
                response.raise_for_status()
                delay = response.elapsed.total_seconds() * 1000
                self.delays.append(delay)
            except requests.RequestException:
                # print(f"请求 {self.url} 时出现错误。")
                self.delays.append(10000)
            # if count > 10:
            #     avg_delay = statistics.mean(self.delays)
            #     std_dev_delay = statistics.stdev(self.delays)
            #     print("avg_delay", avg_delay)
            #     print("std delay", std_dev_delay)
            time.sleep(2)
            # count += 2

    def can_execute_request(self):
        # if None in self.delays:
        #     return False
        avg_delay = statistics.mean(self.delays)
        std_dev_delay = statistics.stdev(self.delays)

        return avg_delay < 300 and std_dev_delay < 50  # 以500ms为基准，但可以根据需要调整


checker = NetworkLatencyChecker()
