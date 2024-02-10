import threading
from src.youtube_video_fetcher import (
    manage_video_urls,
)

thread = threading.Thread(target=manage_video_urls)
thread.daemon = True  # 将线程设置为守护线程，这样当主程序退出时，线程也会退出
thread.start()
