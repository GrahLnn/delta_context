import toml
import os
import yt_dlp
import time
import random
import sys
import threading
from .utils.list_utils import flatten_list
from src.utils.cache_utils import load_channels, load_cache
from tqdm import tqdm
from .utils.LLM_utils import get_completion
from concurrent.futures import ThreadPoolExecutor, as_completed
from .utils.status_utils import countdown
import json
from .utils.video_utils import loggerOutputs

ord_videos = []
ord_urls = []


def extract_url_info(url):
    ydl_opts = {
        "quiet": True,
        # "extract_flat": True,
        # "force_generic_extractor": True,
        # "logger": loggerOutputs,
        # "no_warnings": True,
        # "ignoreerrors": True,  # 忽略错误
    }

    try_count = 0
    while True:
        try:
            if url in ord_urls:
                ordidx = ord_urls.index(url)
                return [url, ord_videos[ordidx]["upload_time"]]
            else:
                # print(f"Extracting info for {url}...")
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    result = ydl.extract_info(url, download=False)
                    # print([url, int(result["upload_date"])], "\n")
                    return [url, int(result["upload_date"])]

        except Exception as e:
            if "This live event will begin in" in str(e):
                return None  # 特定错误，不重试
            elif "Private video." in str(e):
                return None
            elif "Premieres in" in str(e):
                return None
            time.sleep(5)  # 在重试之前等待
            try_count += 1


def process_urls(urls):
    urls_date = []
    with ThreadPoolExecutor(max_workers=30) as executor:
        # 提交任务到线程池
        future_to_url = {executor.submit(extract_url_info, url): url for url in urls}

        # 收集结果
        for future in tqdm(
            as_completed(future_to_url),
            total=len(future_to_url),
            desc="Get video upload date",
        ):
            url_info = future.result()
            if url_info:
                urls_date.append(url_info)

    return urls_date


def get_video_urls(channel_name, choose_type="all"):
    """获取单一频道的所有视频URLs

    prams:
    :type: all, videos ,shorts, streams
    """
    channel_url = f"https://www.youtube.com/{channel_name}"
    ydl_opts = {
        "quiet": True,
        "extract_flat": True,
        "force_generic_extractor": True,
        # "logger": loggerOutputs,
        # "no_warnings": True,
        # "ignoreerrors": True,  # 忽略错误
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(channel_url, download=False)

        urls = []
        if result["entries"][0].get("url", None):
            urls = [entry["url"] for entry in result["entries"]]

        else:
            for item in result["entries"]:
                if (
                    choose_type != "all"
                    and item["webpage_url"] == f"{channel_url}/{choose_type}"
                ):
                    urls.extend([subentry["url"] for subentry in item["entries"]])

                elif (
                    choose_type == "all"
                    and item["webpage_url"] != f"{channel_url}/shorts"
                ):
                    urls.extend([subentry["url"] for subentry in item["entries"]])

        urls_date = process_urls(urls)
        return urls_date


def is_file_older_than_one_day(filepath):
    if not os.path.exists(filepath):
        return True
    last_modified_time = os.path.getmtime(filepath)
    current_time = time.time()
    return (current_time - last_modified_time) > (24 * 60 * 60)


def get_playlists(uploader_id):
    # 创建 yt-dlp 的 YoutubeDL 对象
    ydl_opts = {
        "skip_download": True,  # 不下载视频
        "extract_flat": True,  # 仅提取播放列表概要
        # "ignoreerrors": True,  # 忽略错误
        "quiet": True,  # 减少输出
        # "no_warnings": True,  # 忽略警告
        "logger": loggerOutputs,  # 静默日志
    }
    playlist4videos = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # 尝试获取上传者的所有播放列表
        try:
            result_playlist = ydl.extract_info(
                f"https://www.youtube.com/{uploader_id}/playlists",
                download=False,
            )
            # save_cache(result_playlist, "cache/channel_info.toml")

            # 遍历并打印出所有找到的播放列表信息
            # print(result_playlist["entries"])
            def extract_playlist_info(playlist):
                try:
                    # 假设ydl是预先配置好的，如果每个线程需要独立实例，则应在这里创建
                    result_videos = ydl.extract_info(
                        playlist.get("url"), download=False
                    )
                    videos = [item.get("url") for item in result_videos["entries"]]
                    data = {
                        "title": playlist.get("title"),
                        "url": playlist.get("url"),
                        "id": playlist.get("id"),
                        "videos": videos,
                    }
                    return data
                except Exception as e:
                    print(f"Error extracting playlist {playlist.get('title')}: {e}")
                    return None

            with ThreadPoolExecutor(max_workers=5) as executor:
                # 创建并发任务
                future_to_playlist = {
                    executor.submit(extract_playlist_info, playlist): playlist
                    for playlist in result_playlist["entries"]
                }

                # 等待每个任务完成，并收集结果
                for future in as_completed(future_to_playlist):
                    playlist = future_to_playlist[future]
                    try:
                        data = future.result()
                        if data:
                            playlist4videos.append(data)
                    except Exception as e:
                        print(
                            f"Playlist processing failed for {playlist.get('title')}: {e}"
                        )
        except Exception:
            return None
    return playlist4videos


def make_videos(channel, videos_data):
    cur_urls = []
    while True:
        try:
            videos = get_video_urls(channel)
            # print(f"Got {len(videos)} videos for {channel}.")

            break
        except Exception:
            print(f"Failed to get {channel}'s videos, retrying...")
            time.sleep(5)
    for video_info in videos:
        video = video_info[0]
        upload_date = video_info[1]
        cur_urls.append(
            {"url": video, "uploader": channel, "upload_time": upload_date}
        )  # TODO 加入别的信息

    playlists = get_playlists(channel)
    if playlists:
        for playlist in playlists:
            for video in playlist["videos"]:
                for url in cur_urls:
                    if url["url"] == video and url["url"] not in ord_urls:
                        url["playlist"] = playlist["title"]
                        url["playlist_url"] = playlist["url"]

                        if playlist["title"] not in videos_data:
                            if len(playlist["title"]) > 20:

                                def clean_ans(ans: str):
                                    if ans.startswith('"') and ans.endswith('"'):
                                        ans = ans.strip('"')
                                    ans = (
                                        ans.strip("`")
                                        .strip("。")
                                        .replace("《", "")
                                        .replace("》", "")
                                    )
                                    return ans

                                ans = get_completion(
                                    f"用中文精炼`{playlist['title']}`标题并约束在20个字符内"
                                )
                                ans = clean_ans(ans)

                                if len(ans) > 20:
                                    ans = get_completion(
                                        f"用三个中文词语精炼`{ans}`这个标题",
                                    )
                                    ans = clean_ans(ans)

                                if len(ans) > 20:
                                    print("标题过长:", ans)
                                    ans = ""
                            else:
                                ans = playlist["title"]

                            videos_data[playlist["title"]] = ans
                            url["playlist_zh"] = videos_data[playlist["title"]]
                        else:
                            url["playlist_zh"] = videos_data[playlist["title"]]

                        break
                    elif url["url"] == video and url["url"] in ord_urls:
                        ordidx = ord_urls.index(url["url"])
                        curidx = cur_urls.index(url)
                        cur_urls[curidx] = ord_videos[ordidx]

                        break
    return cur_urls


def update_video_urls():
    urls = []

    playlist_info = "cache/playlist_info.json"
    if os.path.exists(playlist_info):
        with open(playlist_info, "r", encoding="utf-8") as json_file:
            videos_data = json.load(json_file)
    else:
        videos_data = {}
    channels_data = load_channels()
    print(f"Updating video URLs for {len(channels_data)} channels...")

    # 使用线程池处理每个频道
    try:

        with ThreadPoolExecutor(max_workers=1) as executor:
            # 将每个频道的处理任务提交给线程池
            future_to_channel = {
                executor.submit(make_videos, channel, videos_data): channel
                for channel in channels_data.keys()
            }
            print(len(future_to_channel))

            # 通过 tqdm 创建进度条
            for future in tqdm(
                as_completed(future_to_channel),
                total=len(future_to_channel),
                desc="Get video url",
            ):
                channel = future_to_channel[future]
                try:
                    cur_urls = future.result()
                    urls.extend(cur_urls)
                except Exception as e:
                    print(f"Error processing channel {channel}: {e}")
    except KeyboardInterrupt:
        print("KeyboardInterrupt: Exiting...")
        sys.exit(0)

    urls = flatten_list(urls)
    with open(playlist_info, "w", encoding="utf-8") as json_file:
        json.dump(videos_data, json_file, indent=4)
    return urls


def manage_video_urls():
    global ord_videos
    global ord_urls
    if os.path.exists("cache/channels_video.toml"):
        ord_videos = load_channels("cache/channels_video.toml")["videos"]
        ord_urls = [item["url"] for item in ord_videos]
    else:
        ord_videos = []
        ord_urls = []
    while True:
        if not os.path.exists(
            "cache/channels_video.toml"
        ) or is_file_older_than_one_day("cache/channels_video.toml"):
            urls = update_video_urls()
            with open("cache/channels_video.toml", "w") as toml_file:
                data = {"videos": urls}
                toml.dump(data, toml_file)
            countdown(60 * 60 * 24, "Video URLs updated.", "update again.")

        else:
            countdown(60 * 60 * 24, "Video URLs are up to date.", "check again.")


def get_all_video_urls(force_update=False):
    """获取所有频道的视频URLs"""

    urls = []
    cache_file = "cache/channels_video.toml"

    if os.path.exists(cache_file) and not force_update:
        data = load_channels(cache_file)
        urls = data["videos"]
    else:
        print("Updating video URLs...")
        urls = update_video_urls()

        with open(cache_file, "w") as toml_file:
            data = {"videos": urls}
            toml.dump(data, toml_file)

    sorted_urls = sorted(urls, key=lambda x: x["upload_time"], reverse=True)

    return sorted_urls


def filter_items(items):
    new_items = items[:]  # 创建原始 URL 列表的副本

    if os.path.exists("cache/completed_videos.toml"):
        completed_videos = load_cache("cache/completed_videos.toml")["completed_videos"]

        # check_urls = [item.get("url") for item in completed_videos]
        new_items = [
            item for item in new_items if item.get("url") not in completed_videos
        ]
    if os.path.exists("cache/ignore_videos.toml"):
        ignore_videos = load_cache("cache/ignore_videos.toml")["ignore_videos"]
        new_items = [item for item in new_items if item.get("url") not in ignore_videos]
    if os.path.exists("cache/un_complete_video.toml"):
        un_complete_video = load_cache("cache/un_complete_video.toml")[
            "un_complete_video"
        ]
        # print(un_complete_video)
        # check_urls = [item.get("url") for item in un_complete_video]
        new_items = [
            item for item in new_items if item.get("url") not in un_complete_video
        ]
        new_items = [un_complete_video] + new_items
        # print(new_items[0])

    print(f"There are {len(new_items)} videos to be processed.")
    return new_items
