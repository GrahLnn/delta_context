import toml, os, subprocess, time, datetime, tomllib, json
from src.utils.cache_utils import load_cache, save_cache
from threading import Thread
from .LLM_utils import get_completion
from .list_utils import flatten_list


def load_video_info(cache_path, file_id, file_path, file_name, state=True):
    with open(f"{cache_path}/{file_id}.toml", "rb") as toml_file:
        video_info = tomllib.load(toml_file)

    title = video_info["normalized_title"]
    output_video_path = (
        f"{file_path}/{file_name}_zh.mp4" if state else f"{file_path}/{file_name}.mp4"
    )
    thumbnail_path = f"{cache_path}/{file_id}_thumbnail.png"
    tag = video_info["key_words"]
    desc = video_info["normalized_description"]
    season = f"{video_info['playlist_zh']}" if video_info.get("playlist") else ""
    return title, output_video_path, thumbnail_path, tag, desc, season


def make_delivery_info(title, video_path, thumbnail_path, tags, desc, tid, url, season):
    translate_tags = []
    tags = tags.split(",")
    for tag in tags:
        prompt = f"""
用中文翻译`{tag.strip()}`并以如下JSON格式返回
{{
    "tag": [你的翻译]
}}"""
        # print(prompt)
        ans = get_completion(prompt, json_output=True)
        json_ans: dict = json.loads(ans)
        translate_tags.append(list(json_ans.values())[0])

    return {
        "Title": title,
        "Url": url,
        "Video": video_path,
        "Thumbnail": thumbnail_path,
        "Tag": flatten_list(translate_tags),
        "Description": desc,
        "Tid": tid,
        "Season": season,
    }


def read_error_output(process):
    for line in iter(process.stderr.readline, ""):
        print(line, end="")


def deliver_video():
    process = subprocess.Popen(
        ["go", "run", "src/deliver/main.go"],
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    # 使用线程读取标准错误
    stderr_thread = Thread(target=read_error_output, args=(process,))
    stderr_thread.start()

    # 主线程等待子进程结束
    exit_status = process.wait()

    # 等待标准错误线程结束
    stderr_thread.join()
    if exit_status != 0:
        print("Go program exited with an error.")
        # 这里可以执行错误处理
        raise RuntimeError("Go program failed.")


def save_completion(item, completion_path="cache/completed_videos.toml"):
    if not os.path.exists(completion_path):
        completed_videos = []
        # os.makedirs("cache")
    else:
        completed_videos = load_cache(completion_path)["completed_videos"]
    completed_videos.append(item["Url"])
    if completed_videos:
        with open(completion_path, "w") as toml_file:
            data = {"completed_videos": completed_videos}
            toml.dump(data, toml_file)


def interval_deliver():
    # 计算剩余时间
    interval = datetime.timedelta(hours=6)
    # if elapsed_time < interval:
    #     remaining_time = interval - elapsed_time

    #     # 持续显示剩余时间直至时间耗尽
    #     while remaining_time.seconds > 0:
    #         minutes, seconds = divmod(remaining_time.seconds, 60)
    #         formatted_time = f"{minutes:02}:{seconds:02}"
    #         print(
    #             f"Waiting for {formatted_time} before processing the next item.",
    #             end="\r",
    #             flush=True,
    #         )

    #         time.sleep(1)
    #         remaining_time -= datetime.timedelta(seconds=1)
    remaining_time = interval
    while remaining_time.seconds > 0:
        minutes, seconds = divmod(remaining_time.seconds, 60)
        formatted_time = f"{minutes:02}:{seconds:02}"
        print(
            f"Waiting for {formatted_time} before processing the next item.",
            end="\r",
            flush=True,
        )

        time.sleep(1)
        remaining_time -= datetime.timedelta(seconds=1)

    # 打印换行以确保计时结束后的输出不会被上一个输出覆盖
    print()
