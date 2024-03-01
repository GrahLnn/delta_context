import time, datetime, threading, re
import tqdm
from zoneinfo import ZoneInfo


def status_circle(flag, message):
    if isinstance(message, str):
        message = [message]
    chars = ["\uEE06", "\uEE07", "\uEE08", "\uEE09", "\uEE0A", "\uEE0B"]
    i = 0
    elapsed_time = datetime.timedelta(seconds=0)
    start_time = datetime.datetime.now()
    while flag[0]:
        now = datetime.datetime.now()
        elapsed_time = now - start_time
        minutes, seconds = divmod(elapsed_time.seconds, 60)

        # Format the time as MM:SS
        formatted_time = f"{minutes:02}:{seconds:02}"

        print(
            f'{message[0]+" " if message[0] else ""}{chars[i]} {formatted_time}',
            end="          \r",
            flush=True,
        )

        i = (i + 1) % 6
        time.sleep(0.1)

    print(
        f"{message[0]}!",
        end="                 \n",
        flush=True,
    )


def countdown(t, desc="", next_operation=""):
    while t:
        days, remainder = divmod(t, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        timer = f"{desc} Waiting for {days}d:{hours:02}h:{minutes:02}m:{seconds:02}s to {next_operation}          "
        print(timer, end="\r")
        time.sleep(1)
        t -= 1
    # print("Countdown finished!")


def sleep_until_morning(timezone_str="Asia/Shanghai"):
    timezone = ZoneInfo(timezone_str)
    now = datetime.datetime.now(timezone)
    today = now.date()
    start_sleep_time = datetime.datetime.combine(
        today, datetime.time(23, 0), tzinfo=timezone
    )
    end_sleep_time = datetime.datetime.combine(
        today, datetime.time(8, 0), tzinfo=timezone
    ) + datetime.timedelta(days=1)

    if now < datetime.datetime.combine(today, datetime.time(8, 0), tzinfo=timezone):
        end_sleep_time -= datetime.timedelta(days=1)
    # elif now > datetime.datetime.combine(today, datetime.time(23, 0), tzinfo=timezone):
    #     start_sleep_time += datetime.timedelta(days=1)

    if start_sleep_time <= now <= end_sleep_time:
        sleep_seconds = int((end_sleep_time - now).total_seconds())
        countdown(sleep_seconds, "Sleeping", "morning")


def print_status(flag, desc=""):
    state_thread = threading.Thread(target=status_circle, args=(flag, desc))
    state_thread.start()


def get_seconds(time):
    h, m, s, ms = map(int, re.split("[:.]", time))
    return h * 3600 + m * 60 + s + ms / 1000


def create_progress_bar(progress, total_length=100):
    # 将进度转换为基于 total_length 的值，确保小数进度得到反映
    if progress > 1:
        return "radio must be less than 1."
    full_blocks = int(progress * total_length)
    completed = "\uEE04" * (full_blocks - 1)  # 完成的块

    # 处理进度条的开始和结束字符
    if full_blocks > 0:
        completed = "\uEE03" + completed  # 添加开始字符
    else:
        completed = "\uEE00"  # 如果没有完成的块，则以开始字符开头

    # 确保总长度为 total_length
    incompleted = "\uEE01" * (total_length - full_blocks - 1)  # 未完成的块
    if full_blocks == total_length or progress > 0.99:
        completed += "\uEE05"  # 当进度为 100% 时，添加结束字符
    else:
        incompleted += "\uEE02"  # 如果未完成，则添加结束字符

    return completed + incompleted


def progress_bar_handler(progress, spinner_active, task_name=""):
    spinner = ["\uEE06", "\uEE07", "\uEE08", "\uEE09", "\uEE0A", "\uEE0B"]
    spinner_idx = 0
    while spinner_active[0]:
        current_progress = progress[0]
        print(
            f"\r{task_name} {create_progress_bar(current_progress/100)} {spinner[spinner_idx]} {current_progress:.2f}%",
            end="  ",
        )
        spinner_idx = (spinner_idx + 1) % len(spinner)
        time.sleep(0.1)
    current_progress = progress[0]
    print(
        f"\r{task_name} {create_progress_bar(current_progress/100)}  {current_progress:.2f}%",
    )


class _CustomProgressBar(tqdm.tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current = self.n  # Set the initial value

    def update(self, n):
        super().update(n)
        self._current += n

        # 计算当前的进度比例
        progress_ratio = self._current / self.total
        # 使用 create_progress_bar 函数生成进度条
        progress_bar = create_progress_bar(progress_ratio)

        # 打印进度条

        print(
            "Transcribing: "
            + progress_bar
            + " "
            + str(self._current)
            + "/"
            + str(self.total)
            + f"|{round(progress_ratio * 100, 2)}%",
            end=" \r",
            flush=True,
        )
