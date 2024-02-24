import os, yt_dlp, requests, shutil, subprocess, threading, sys, time, string, re, json
from urllib.parse import urlparse
from .LLM_utils import get_completion
from .file_utils import sanitize_filename, url_type, save_to_toml
from .cache_utils import load_cache, save_cache
from .text_utils import strip_chinese_punctuation
from asset.env.env import ADD_TRANSITION, target_language
from .image_utils import resize_img
from moviepy.editor import VideoFileClip, concatenate_videoclips
from src.utils.status_utils import (
    print_status,
    get_seconds,
    create_progress_bar,
    progress_bar_handler,
)
import cv2


def fetch_video_info(url, max_retries=3):
    options = {
        "quiet": True,
        "extract_flat": True,
        "force_generic_extractor": True,
    }

    attempts = 0
    while attempts < max_retries:
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                info_dict = ydl.extract_info(url, download=False)

                title = info_dict.get("title", None)
                video_url = info_dict.get("webpage_url", None)
                description = info_dict.get("description", None)
                uploader = info_dict.get("uploader", None)
                thumbnail = info_dict.get("thumbnail", None)

                return {
                    "title": title,
                    "description": description,
                    "uploader": uploader,
                    "thumbnail": thumbnail,
                    "video_url": video_url,
                }
        except Exception as e:
            print(f"Attempt {attempts + 1} failed with error: {e}")
            attempts += 1

    print("Failed to fetch video info after maximum retries.")
    return None


def download_thumbnail(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
        resize_img(save_path)


def download_mp4(video_url, file_name):
    max_retries = 10  # 设置最大重试次数
    retry_count = 0

    ydl_opts = {
        "format": "bestvideo+bestaudio/best",
        "outtmpl": file_name + ".%(ext)s",
        "postprocessors": [
            {
                "key": "FFmpegVideoConvertor",
                "preferedformat": "mp4",
            }
        ],
    }

    while retry_count < max_retries:
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            break  # 如果下载成功，跳出循环

        except Exception as e:  # 捕获所有可能的异常
            print(f"Error during download: {e}")
            retry_count += 1
            if retry_count < max_retries:
                print(f"Retrying {video_url}... attempt {retry_count}")
                time.sleep(5)  # 等待5秒后再重试

    if retry_count == max_retries:
        print(f"Failed to download after {max_retries} attempts.")


def extract_filename_and_extension(url):
    # 解析URL
    parsed_url = urlparse(url)

    # 从路径中提取文件名
    filename = os.path.basename(parsed_url.path)

    # 分离文件名和文件类型
    base_name, extension = os.path.splitext(filename)

    return base_name.replace("'", "").replace(",", "").strip(), extension


def extract_audio_from_video(video_path, output_audio_path):
    command = ["ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", output_audio_path]
    subprocess.run(command)


def add_extra_info(video_info):
    description_normalized = video_info["description"].replace("\n", " ")
    prompt = f"""
你是一位精通简体中文的专业翻译，尤其擅长将专业学术论文翻译成浅显易懂的形式。
context:
{description_normalized}
Based context information translate following title to Chinese:
{video_info['title']}

然后放在以下JSON格式里：
{{
    "chinese_title":"[title的中文翻译]",
}}"""
    title = get_completion(
        prompt,
        json_output=True,
    )
    title = json.loads(title)["chinese_title"]
    title = (
        strip_chinese_punctuation(title.strip().strip(string.punctuation))
        .replace("《", "")
        .replace("》", "")
    )
    description = get_completion(
        f"对以下内容保留主要信息：\n---\n{video_info['description']}"
    )
    description = get_completion(
        description,
        sys_prompt=f"Translate to {target_language}",
    )
    key_words = get_completion(
        video_info["description"],
        sys_prompt="选择八个词语作为标签并以逗号隔开",
    )
    video_info["normalized_title"] = f"【{title}】【中字】"
    video_info["normalized_description"] = (
        f"{video_info['uploader']}·{video_info['playlist']}\n{video_info['video_url']}\n{description}"
        if video_info.get("playlist")
        else f"{video_info['uploader']}\n{video_info['video_url']}\n{description}"
    )
    video_info["key_words"] = key_words.replace("，", ",")

    return video_info


def get_video_info(video_url, max_retries=15, wait_time=5):
    attempt = 0
    video_info = None

    while attempt < max_retries:
        try:
            with yt_dlp.YoutubeDL({}) as ydl:
                video_info = ydl.extract_info(video_url, download=False)
            break  # 如果成功获取信息，跳出循环
        except yt_dlp.utils.DownloadError as e:
            print(f"Error occurred: {e}; retrying in {wait_time} seconds.")
            time.sleep(wait_time)
            attempt += 1
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break  # 对于非网络错误，可能不宜重试

    return video_info


def handle_remote_video(video):
    video_info = get_video_info(video.get("url"))

    # print(video)
    # print(video_info["playlist"])
    # sys.exit()

    file_name = sanitize_filename(video_info["title"]).strip()
    file_id = video_info["id"]
    file_path = f"video/{file_name}"
    subtitle_path = f"{file_path}/subtitle"
    cache_path = f"{file_path}/cache"
    audio_file = f"{file_path}/{file_id}.wav"
    video_file = f"{file_path}/{file_name}.mp4"
    season = (
        f"{video_info['uploader']}·{video_info['playlist']}"
        if video_info.get("playlist")
        else f"{video_info['uploader']}"
    )
    # print(file_name, file_id, file_path, audio_file, video_file)
    if not os.path.exists(file_path):
        os.makedirs(subtitle_path)
        os.makedirs(cache_path)
    if not os.path.exists(f"{cache_path}/{file_id}.toml"):
        video_info = fetch_video_info(video.get("url"))
        if video.get("playlist"):
            video_info["playlist"] = video.get("playlist")
            video_info["playlist_zh"] = video.get("playlist_zh")
        # print("[error check 1]", video_info)
        if ADD_TRANSITION:
            # print("Adding extra info...")
            video_info = enhance_video_info(video_info)
            # print("[error check 2]", video_info)
        write_video_info(f"{cache_path}/{file_id}", video_info)

    if not os.path.exists(video_file):
        start_mp4_download(video.get("url"), f"{file_path}/{file_name}")

    if not os.path.exists(audio_file):
        download_audio(video.get("url"), f"{file_path}/{file_id}")
    else:
        print(f"'{file_name}' already exists")
    return file_name, file_id, file_path, audio_file, cache_path, subtitle_path, season


def handle_local_video(video_url):
    file_name, ext = extract_filename_and_extension(video_url)
    if ext:
        save_path = f"video/{file_name}"
        subtitle_path = f"{save_path}/subtitle"
        cache_path = f"{save_path}/cache"
        file_path = f"{save_path}/{file_name}{ext}"
        audio_file = f"{save_path}/{file_name}.wav"
        file_id = file_name

        if not os.path.exists(save_path):
            os.makedirs(save_path)
            os.makedirs(subtitle_path)
            os.makedirs(cache_path)

        if not os.path.exists(file_path):
            shutil.copyfile(video_url, file_path)

        if not os.path.exists(audio_file):
            extract_audio_from_video(file_path, audio_file)
    else:
        print("Invalid video URL, please check again and retry.")
    return file_name, file_id, file_path, audio_file, cache_path, subtitle_path


def enhance_video_info(video_info):
    if not video_info.get("normalized_title", "") or not video_info.get(
        "normalized_description", ""
    ):
        # print("Adding extra info...")
        video_info = add_extra_info(video_info)

    return video_info


threads = []


def start_mp4_download(video_url, output_path):
    global threads
    download_thread = threading.Thread(
        target=download_mp4, args=(video_url, output_path)
    )
    threads.append(download_thread)
    print(len(threads), "threads are running")
    print()
    download_thread.start()


def download_audio(video_url, output_path):
    ydl_opts = {
        "format": "m4a/bestaudio/best",
        "outtmpl": output_path + ".%(ext)s",
        "retries": 10,  # 无限重试
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])


def video_file_info(video):
    if url_type(video.get("url")) == "remote":
        return handle_remote_video(video)
    elif url_type(video.get("url")) == "local":
        return handle_local_video(video)
    else:
        print(f"Invalid file or URL: '{video.get('url')}'")


def insert_subtitle(
    input_video_path, subtitle_path, file_path, file_id, file_name, max_retries=3
):
    try:
        cmd = [
            "ffmpeg",
            "-i",
            input_video_path,  # 输入视频文件
            "-vf",
            f"subtitles={subtitle_path}/{file_id}_zh.ass",  # 设置字幕文件
            "-c:a",
            "copy",  # 不重新编码音频
            "-y",  # 覆盖输出文件（如果存在）
            f"{file_path}/{file_name}.mp4",  # 输出视频文件
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            text=True,
        )

        # while True:
        #     output = process.stdout.readline()
        #     if output == "" and process.poll() is not None:
        #         break
        #     if output:
        #         print(output.strip())

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


def write_video_info(meta_file_path, video_info):
    toml_file_path = f"{meta_file_path}.toml"
    thumbnail_save_path = f"{meta_file_path}_thumbnail.png"

    save_to_toml(video_info, toml_file_path)
    download_thumbnail(video_info["thumbnail"], thumbnail_save_path)


def merge_videos(video_list, output_path):
    """
    合并视频列表中的视频并保存到指定的输出路径。

    :param video_list: 包含视频文件路径的列表。
    :param output_path: 合并后的视频文件的输出路径。
    """
    # 加载视频文件
    clips = [VideoFileClip(video) for video in video_list]

    # 合并视频
    final_clip = concatenate_videoclips(clips)

    # 输出视频
    final_clip.write_videofile(output_path, codec="libx264")


def get_video_aspect_ratio(video_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    # 获取视频的宽度和高度
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # 计算长宽比
    aspect_ratio = width / height

    # 释放视频文件
    cap.release()

    return aspect_ratio
