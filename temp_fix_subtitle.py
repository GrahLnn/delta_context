import re
import subprocess
import threading
import time
from src.utils.status_utils import (
    get_seconds,
    progress_bar_handler,
)


def insert_subtitle(ord_v, sub_v, out_v):
    try:
        cmd = [
            "ffmpeg",
            "-i",
            ord_v,  # 输入视频文件
            "-vf",
            f"subtitles={sub_v}",  # 设置字幕文件
            "-c:a",
            "copy",  # 不重新编码音频
            "-y",  # 覆盖输出文件（如果存在）
            out_v,  # 输出视频文件
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            text=True,
        )
        duration = None
        progress = [0]
        active = [True]

        # 创建并启动进度条线程
        time.sleep(1)
        progress_thread = threading.Thread(
            target=progress_bar_handler, args=(progress, active, "Insert subtitle")
        )
        progress_thread.start()
        for line in process.stdout:
            if duration is None:
                match = re.search(r"Duration: (\d{2}:\d{2}:\d{2}\.\d{2}),", line)
                if match:
                    duration = get_seconds(match.group(1))

            match = re.search(r"time=(\d{2}:\d{2}:\d{2}\.\d{2})", line)
            if match:
                elapsed_time = get_seconds(match.group(1))
                if duration:
                    progress[0] = (elapsed_time / duration) * 100

        process.wait()
        active[0] = False
        progress_thread.join()
    except Exception as e:
        active[0] = False
        raise e


insert_subtitle()
