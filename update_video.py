from src.youtube_video_fetcher import (
    manage_video_urls,
)

from src.utils.net_utils import checker


try:
    checker.can_execute_request()
except Exception:
    checker.start()
manage_video_urls()
