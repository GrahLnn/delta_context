import time, random, asyncio, shutil, os, tomllib, librosa, threading, spacy, sys
from src.youtube_video_fetcher import (
    get_all_video_urls,
    filter_items,
    manage_video_urls,
)
from src.utils.video_utils import insert_subtitle
from src.main_auxiliary_utils import (
    filter_channel,
    add_addition_videos,
    get_deliver_video,
    get_transcribe,
    align_due_sentences,
    chunk_texts,
    save_subtitle,
    split_audio,
    split_video,
    clean_vocal,
    process_task,
    get_cut_stamps,
    download_extract,
    waiting_thread,
    deliver_and_save_completion,
    comment_tasks,
)
from src.env.env import LIMIT_LEN_TIME
from src.utils.align_utils import align_transcripts
from src.utils.examine_utils import rebuild_result_sentences
from src.utils.cache_utils import get_or_cache, load_cache, save_cache
from src.utils.net_utils import checker
from pydub import AudioSegment

# from src.utils.deliver_utils import interval_deliver
from src.utils.LLM_utils import cost_calculator
from src.utils.video_utils import threads, merge_videos, get_video_aspect_ratio
from src.utils.transcribe_utils import merge_chunks_for_check, merge_chunks
from src.utils.file_utils import sorted_alphanumeric, delete_contents_of_directory

# from src.utils.bilibili_utils import get_credential
from src.utils.audio_utils import calculate_energy
import datetime
from redlines import Redlines
import json
from src.utils.summary_utils import get_timed_texts, get_summary_text
from src.utils.video_comment import daemon_thread
import httpx

# from bilibili_api import settings

# settings.proxy = "https://api.mahiron.moe"
checker.start()
limit = None
addition = [
    {
        "url": "https://www.youtube.com/watch?v=jrVQXHmxH7Y",
        "uploader": "@TheRoyalInstitution",
    }
]
channel_filter = []
playlist_filter = ["Flutter Package of the Week", "Flutter Widget of the Week"]

video_datas = get_all_video_urls(force_update=False)

video_datas = filter_channel(channel_filter, video_datas)
# fileter playlist
video_datas = add_addition_videos(addition, video_datas)
video_datas = filter_items(video_datas)
video_datas = (
    [item for item in video_datas if item.get("playlist") not in playlist_filter]
    if playlist_filter
    else video_datas
)
capta_server = "http://127.0.0.1:5000/detect"
try:
    httpx.post(capta_server)
except Exception as e:
    print(e)
    print("capta server not start")

# sys.exit()
# thread = threading.Thread(target=manage_video_urls)
# thread.daemon = True  # 将线程设置为守护线程，这样当主程序退出时，线程也会退出
# thread.start()

video_datas = video_datas[:limit] if limit else video_datas
# sys.exit()
# credential = get_credential()

if os.path.exists("cache/comment_task.toml"):
    with open("cache/comment_task.toml", "rb") as toml_file:
        pre_comment_task = tomllib.load(toml_file)["comment_task"]
    comment_tasks.extend(pre_comment_task)

daemon = threading.Thread(target=daemon_thread, args=(comment_tasks,), daemon=True)
daemon.start()

if os.path.exists("cache/delivery_videos.toml"):
    with open("cache/delivery_videos.toml", "rb") as toml_file:
        delivery_videos = tomllib.load(toml_file)["delivery_videos"]
    deliver_and_save_completion(delivery_videos)
    video_datas = [
        item for item in video_datas if item.get("url") not in delivery_videos
    ]

    # asyncio.run(deliver_and_save_completion(delivery_videos, credential))
    # interval_deliver()


# TODO 实现全局一致的名词翻译，多说话人盲源分离
# interval_deliver()


for item in video_datas:
    start_time = datetime.datetime.now()
    elapsed_time = datetime.timedelta(seconds=0)
    print(item)
    while True:
        try:
            (
                fname,
                fid,
                fpath,
                afile,
                cpath,
                sbpath,
                uploader,
                clip_path,
                season,
            ) = download_extract(item)
            break
        except Exception:
            print("download error")
            time.sleep(10)
    print(fname)
    audio = AudioSegment.from_file(afile)
    audio_length_minutes = len(audio) / (1000 * 60)  # 长度转换为分钟

    info_path = f"{cpath}/{fid}.toml"
    info = load_cache(f"{cpath}/{fid}.toml")

    process_task(
        info,
        "vocal_is_clean",
        info_path,
        clean_vocal,
        afile,
        fpath,
        fid,
    )
    audio, sr = librosa.load(afile, sr=None)
    energy_threshold = 0.0005
    energy = calculate_energy(audio)
    print(f"Audio energy: {energy:.6f}")
    # sys.exit()
    # result = (
    #     get_transcribe(afile, cpath, add_timestamps=True)
    #     if energy > energy_threshold
    #     else {"text": ""}
    # )
    result = get_transcribe(afile, cpath, add_timestamps=True)
    words = []
    for chunk in result["chunks"]:
        words.extend(chunk["words"])

    # sys.exit()
    if len(result["text"].split()) < 50:
        videos = get_deliver_video(
            cpath, fid, fpath, fname, uploader, item.get("url"), False
        )

        waiting_thread(threads)
        comment_task = deliver_and_save_completion(videos)
        # asyncio.run(deliver_and_save_completion(videos, credential))

    else:
        # if audio_length_minutes > LIMIT_LEN_TIME:
        #     chunk_check = get_or_cache(
        #         f"{cpath}/chunk_check.toml",
        #         lambda: {"chunks": merge_chunks_for_check(result["chunks"])},
        #     )
        #     cut_stamps = get_cut_stamps(
        #         chunk_check["chunks"]
        #     )  # 分割视频，result，若是有修改term的时候怎么办\
        #     cut_texts = merge_chunks(cut_stamps, result["chunks"])
        #     save_cache({"cut_texts": cut_texts}, f"{cpath}/cut_texts.toml")

        #     big_chunks = [chunk_texts(text) for text in cut_texts]
        #     save_cache({"big_chunks": big_chunks}, f"{cpath}/big_chunks.toml")
        #     # print(len(cut_texts), len(cut_stamps) + 1)
        #     # print(cut_stamps)
        #     sys.exit()
        #     process_task(
        #         info,
        #         "audio_is_split",
        #         info_path,
        #         split_audio,
        #         cut_stamps,
        #         afile,
        #         clip_path,
        #     )
        #     video_path = f"{fpath}/{fname}.mp4"
        #     waiting_thread(threads)
        #     process_task(
        #         info,
        #         "video_is_split",
        #         info_path,
        #         split_video,
        #         cut_stamps,
        #         video_path,
        #         clip_path,
        #     )
        #     # print(cut_stamps)

        # elif audio_length_minutes <= LIMIT_LEN_TIME and audio_length_minutes > 2:
        if audio_length_minutes > 2:
            shutil.copy(afile, f"{clip_path}/audio")
            waiting_thread(threads)
            shutil.copy(f"{fpath}/{fname}.mp4", f"{clip_path}/video")
            big_chunks = [chunk_texts(result["text"])]
            save_cache({"big_chunks": big_chunks}, f"{cpath}/big_chunks.toml")

            words = []
            for chunk in result["chunks"]:
                words.extend(chunk["words"])
            print(
                "len for transcript",
                len(result["text"].split()),
                len([word["word"] for word in words]),
            )
            # test = Redlines(result["text"], "".join([word["word"] for word in words]))

            # with open(f"{cpath}/redlines.md", "w", encoding="utf-8") as f:
            #     f.write(test.output_markdown)
            # sys.exit()

        else:
            if os.path.exists("cache/ignore_videos.toml"):
                ignore_videos = load_cache("cache/ignore_videos.toml")["ignore_videos"]
                ignore_videos.append(item.get("url"))
                save_cache({"ignore_videos": ignore_videos}, "cache/ignore_videos.toml")
            else:
                save_cache(
                    {"ignore_videos": [item.get("url")]}, "cache/ignore_videos.toml"
                )

            continue
        # chunk_check = get_or_cache(
        #     f"{cpath}/chunk_check.toml",
        #     lambda: {"chunks": merge_chunks_for_check(result["chunks"])},
        # )
        audio_clips = os.listdir(f"{clip_path}/audio")
        audio_clips = (
            sorted_alphanumeric(audio_clips) if len(audio_clips) > 1 else audio_clips
        )
        audio_clips_path = [f"{clip_path}/audio/{file}" for file in audio_clips]
        # print(audio_clips)
        video_clips = os.listdir(f"{clip_path}/video")
        video_clips = (
            sorted_alphanumeric(video_clips) if len(video_clips) > 1 else video_clips
        )
        video_clips_path = [f"{clip_path}/video/{file}" for file in video_clips]
        # sys.exit()
        aspect_ratio = get_video_aspect_ratio(video_clips_path[0])
        for idx, (audio, video, chunk) in enumerate(
            zip(audio_clips_path, video_clips_path, big_chunks)
        ):
            clip_cpath = f"{cpath}/{idx+1}" if len(audio_clips) > 1 else cpath
            if not os.path.exists(clip_cpath):
                os.mkdir(clip_cpath)
            try:
                seg_transcripts, seg_translates, words = align_due_sentences(
                    chunk, clip_cpath, words
                )  # 通过之前摘取的result，来提取seg部分
            except Exception:
                save_cache({"un_complete_video": item}, "cache/un_complete_video.toml")
                raise Exception("Un complete video")

            # result = align_transcripts(
            #     seg_transcripts, audio, clip_cpath
            # )  # 然后对每一个align结果并加上cut时间
            for iidx, word in enumerate(words):
                words[iidx]["word"] = word["word"].strip()
                words[iidx]["start"] = float(word["start"])
                words[iidx]["end"] = float(word["end"])

            result = {"word_segments": words}
            print("aspect_ratio:", aspect_ratio)
            result = rebuild_result_sentences(
                result, seg_transcripts, seg_translates
            )  # TODO 验证句子是否横跨check result里的项
            # print(result["segments"][0])
            if aspect_ratio < 1.5:

                for idx, seg in enumerate(result["segments"]):
                    if len(result["segments"][idx]["translation"]) > 18:
                        check_sentences = result["segments"][idx]["translation"].split(
                            " "
                        )
                        nlp = spacy.load("zh_core_web_sm")
                        new_sentences = []
                        for sent in check_sentences:
                            if len(sent) > 18:

                                # 用spacy对每个句子进行分词，然后组装出新的句子使得少于18个字符，然后对新的所有句子用join组装成新的句子，用空格分开
                                doc = nlp(sent)
                                cur_sentence = ""
                                for token in doc:
                                    if len(cur_sentence) + len(token.text) > 18:
                                        new_sentences.append(cur_sentence)
                                        cur_sentence = token.text
                                    else:
                                        cur_sentence += token.text
                                new_sentences.append(cur_sentence)
                            else:
                                new_sentences.append(sent)
                        result["segments"][idx]["translation"] = " ".join(
                            new_sentences
                        ).strip()

            subtitle_path = f"{sbpath}/{idx+1}" if len(audio_clips) > 1 else sbpath
            if not os.path.exists(subtitle_path):
                os.mkdir(subtitle_path)
            save_subtitle(subtitle_path, audio, result)
            waiting_thread(threads)
            clip_id = f"{idx+1}" if len(audio_clips) > 1 else fid
            video_name = clip_id if len(audio_clips) > 1 else fname
            # TODO 验证视频长宽比来调整空格位置（18）

            process_task(
                info,
                f"{video_clips[idx]}_is_insert_subtitle",
                info_path,
                insert_subtitle,
                video,
                subtitle_path,
                f"{clip_path}/translate_video",
                clip_id,
                video_name,
            )
            subtitle_json_file = f"subtitles={subtitle_path}/{clip_id}.json"
            subtitle_json = json.load(open(subtitle_json_file, "r"))
            timed_texts = get_timed_texts(subtitle_json)
            summary = asyncio.run(get_summary_text(timed_texts))
            info["summary"] = summary
            save_cache(info, info_path)
            print(
                (
                    f"Done processing clip {idx+1} of {len(audio_clips)}. Starting next clip."
                    if idx + 1 < len(audio_clips)
                    else f"Done processing clip {idx+1} of {len(audio_clips)}. Now merge clips from {fname}."
                )
                if len(audio_clips) > 1
                else f"Done processing {item.get('url')}."
            )

        # sys.exit()
        # 这里要合并视频
        translate_videos = os.listdir(f"{clip_path}/translate_video")
        translate_videos = (
            sorted_alphanumeric(translate_videos)
            if len(translate_videos) > 1
            else translate_videos
        )
        # print(clip_path)
        # print(translate_videos)
        video_clips_path = [
            f"{clip_path}/translate_video/{file}" for file in translate_videos
        ]
        if len(video_clips_path) > 1:
            process_task(
                info,
                "video_is_merge",
                info_path,
                merge_videos,
                video_clips_path,
                f"{fpath}/{fname}_zh.mp4",
            )
        elif video_clips_path:
            shutil.copy(video_clips_path[0], f"{fpath}/{fname}_zh.mp4")
        # sys.exit()
        videos = get_deliver_video(
            cpath, fid, fpath, fname, uploader, item.get("url"), summary=summary
        )  # TODO: 判断分区字符数描述限制

        # print(videos)
        # sys.exit(1)
        deliver_and_save_completion(videos)

        # asyncio.run(deliver_and_save_completion(videos, credential))
        delete_contents_of_directory(f"{clip_path}/translate_video")
        if os.path.exists("cache/un_complete_video.toml"):
            os.remove("cache/un_complete_video.toml")

    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time.seconds, 60)
    formatted_time = f"{minutes:02}:{seconds:02}"
    time.sleep(0.5)
    print(
        f"Done processing `{fid}` from `{item.get('uploader')}`, a {round(audio_length_minutes,2)} minutes video, take {formatted_time} minute and ${round(cost_calculator.get_total_cost(),3)} API call to process."
    )
    cost_calculator.reset_total_cost()
    # interval_deliver()
