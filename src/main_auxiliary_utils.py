import toml, os, spacy, tomllib, time, asyncio, copy, sys
from src.utils.transcribe_utils import (
    transcribe_with_distil_whisper,
    merge_if_not_period,
    find_closest_stamps,
)
from src.utils.translate_utils import ask_LLM_to_translate

from src.utils.trimming_sentence_utils import (
    concatenate_sentences_punc_end,
    split_stage_one,
    split_stage_two,
    separate_too_long_tl,
    clean_sentences,
)

from src.utils.subtitle_utils import get_writer
from src.utils.cache_utils import save_cache, load_channels, get_or_cache, load_cache
from src.utils.video_utils import video_file_info
from src.utils.status_utils import print_status
from src.utils.audio_utils import extract_vocal
from src.utils.deliver_utils import (
    load_video_info,
    make_delivery_info,
    save_completion,
)

# from src.utils.LLM_utils import cost_calculator
from src.utils.bilibili_utils import upload
from src.utils.video_comment import video_info
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
from datetime import datetime, timedelta
from redlines import Redlines
import re
import string
from bilibili_api import sync


def download_extract(item):
    flag = [True]
    # message = ["download video"]
    try:
        # print_status(flag, message)
        _, uploader = item.get("url"), item.get("uploader")
        fname, fid, fpath, afile, cpath, sbpath, season = video_file_info(item)
        if not os.path.exists(f"{fpath}/clip"):
            os.makedirs(f"{fpath}/clip/audio")
            os.makedirs(f"{fpath}/clip/video")
            os.makedirs(f"{fpath}/clip/translate_video")
        clip_path = f"{fpath}/clip"
        flag[0] = False
        # print(fname, fid, fpath, afile, cpath, sbpath, uploader)
        return fname, fid, fpath, afile, cpath, sbpath, uploader, clip_path, season
    except Exception as e:
        print(e)
        flag[0] = False


def clean_vocal(afile, output_path, output_name):
    flag = [True]
    # message = ["clean vocal"]
    try:
        # print_status(flag, message)
        extract_vocal(afile, output_path, output_name)
        flag[0] = False
        print()
    except Exception as e:
        print(e)
        flag[0] = False


def get_transcribe(audio_file, cache_path, add_timestamps=False):
    result = get_or_cache(
        f"{cache_path}/transcribe_result.toml",
        lambda: transcribe_with_distil_whisper(audio_file, add_timestamps),
    )

    return result


def chunk_texts(text):
    def join_sentences(sentences, group_size):
        # 为了安全起见，我们先把句子转换成字符串
        sentences = [str(sentence) for sentence in sentences]

        return [
            " ".join(sentences[i : i + group_size])
            for i in range(0, len(sentences), group_size)
        ]

    nlp_en = spacy.load("en_core_web_sm")

    doc = nlp_en(text)
    sentences = [sent.text for sent in doc.sents]
    sentences = concatenate_sentences_punc_end(sentences)

    grouped_sentences = join_sentences(sentences, 10)
    return grouped_sentences


def align_due_sentences(texts, cache_path, words):
    def check_words_equal(ord_texts, proc_texts, words):
        test = Redlines(ord_texts, proc_texts, markdown_style="none")
        text = test.output_markdown
        tags = [
            (m.start(), m.group())
            for m in re.finditer(r"<del>.*?</del>|<ins>.*?</ins>", text)
        ]
        # print(tags)

        # 初始化del标签的计数器
        del_count = 0

        # ins_str = {"follow_del_pos": 0, "ins_text": "", "ins_lenth": 0}

        # 遍历tags，同时跟踪<del>和<ins>标签
        del_info = []
        for _, tag in tags:
            if "<del>" in tag:
                # 计数<del>标签
                del_count += 1

                del_info.append(
                    {
                        "del_text": re.search(r"<del>(.*?)</del>", tag).group(1),
                        "del_count": del_count,
                        "ins_text": None,
                        "ins_lenth": None,
                        "start_idx": None,
                        "end_idx": None,
                    }
                )
            elif "<ins>" in tag:
                # 显示<ins>标签相对于前一个<del>标签的位置

                del_info[-1]["ins_text"] = re.search(r"<ins>(.*?)</ins>", tag).group(1)
                del_info[-1]["ins_lenth"] = len(
                    re.search(r"<ins>(.*?)</ins>", tag).group(1).split()
                )

        # replace所有的<ins>标签
        for info in del_info:
            if info["ins_text"]:
                text = text.replace(
                    f"<ins>{info['ins_text']}</ins>",
                    "",
                )

        split_text = text.split()

        count = 0
        for idx, text in enumerate(split_text):
            if "<del>" in text:

                del_info[count]["start_idx"] = idx
                del_info[count]["end_idx"] = idx + len(
                    del_info[count]["del_text"].split()
                )
                # print(count)

                # _, tag = tags[count]

                if del_info[count]["ins_text"]:
                    ord = [
                        w["word"].strip().lower().strip(string.punctuation)
                        for w in words[
                            del_info[count]["start_idx"] : del_info[count]["end_idx"]
                        ]
                    ]

                    cur = [
                        w.strip().strip(string.punctuation)
                        for w in del_info[count]["ins_text"].lower().split()
                    ]
                    # print(ord)
                    # print(cur)
                    # print(len(a.split()), len(words))

                    diff = Redlines(" ".join(ord), " ".join(cur), markdown_style="none")
                    diff_text = diff.output_markdown
                    # print(diff_text)

                    ins_tags = [
                        (m.start(), m.group())
                        for m in re.finditer(
                            r"<del>.*?</del>|<ins>.*?</ins>", diff_text
                        )
                    ]

                    del_del_info = []
                    for _, tag in ins_tags:
                        if "<del>" in tag:
                            # 计数<del>标签

                            del_del_info.append(
                                {
                                    "del_text": re.search(
                                        r"<del>(.*?)</del>", tag
                                    ).group(1),
                                    "ins_text": None,
                                    "ins_lenth": None,
                                    "start_idx": None,
                                    "end_idx": None,
                                }
                            )
                        elif "<ins>" in tag:
                            # 显示<ins>标签相对于前一个<del>标签的位置

                            del_del_info[-1]["ins_text"] = re.search(
                                r"<ins>(.*?)</ins>", tag
                            ).group(1)
                            del_del_info[-1]["ins_lenth"] = len(
                                re.search(r"<ins>(.*?)</ins>", tag).group(1).split()
                            )
                    inner_count = 0
                    for iex, text in enumerate(diff_text.split()):
                        if "<del>" in text:
                            del_del_info[inner_count]["start_idx"] = iex
                            del_del_info[inner_count]["end_idx"] = iex + len(
                                del_del_info[inner_count]["del_text"].split()
                            )
                            inner_count += 1
                    # for _, tag in ins_tags:
                    #     if "<ins>" in tag:
                    #         raise ValueError(f"ins标签不应该存在 {del_info[count]}")
                    #     elif "<del>" in tag:
                    #         del_text.append(re.search(r"<del>(.*?)</del>", tag).group(1))
                    for info in del_del_info:
                        if info["ins_text"]:
                            diff_text = diff_text.replace(
                                f"<ins>{info['ins_text']}</ins>",
                                "",
                            )
                    for info in del_del_info:
                        if info["ins_text"]:
                            diff_text = diff_text.replace(
                                f"<del>{info['del_text']}</del>",
                                f"{info['del_text']}",
                            )
                            if len(info["ins_text"].split()) == len(
                                info["del_text"].split()
                            ):
                                # print(info["start_idx"], info["end_idx"])
                                # print(
                                #     words[
                                #         del_info[count]["start_idx"]
                                #         + info["start_idx"] : del_info[count]["start_idx"]
                                #         + info["end_idx"]
                                #     ]
                                # )
                                # change_len = info["end_idx"] - info["start_idx"]
                                insert_words = info["ins_text"].split()
                                start_idx = (
                                    del_info[count]["start_idx"] + info["start_idx"]
                                )
                                end_idx = del_info[count]["start_idx"] + info["end_idx"]
                                for i in range(start_idx, end_idx):
                                    words[i]["word"] = insert_words[i - start_idx]
                                # print(
                                #     words[
                                #         del_info[count]["start_idx"]
                                #         + info["start_idx"] : del_info[count]["start_idx"]
                                #         + info["end_idx"]
                                #     ]
                                # )
                            else:
                                raise ValueError(
                                    f"插入和删除的长度不一致 {del_info[count]} {info}"
                                )

                    # del_text = [w["del_text"] for w in del_del_info]
                    ins_list = del_info[count]["ins_text"].split()

                    inner_count = 0
                    ins_inner_count = 0
                    for iex, text in enumerate(diff_text.split()):
                        if "<del>" in text:
                            cur_item_len = len(
                                del_del_info[inner_count]["del_text"].split()
                            )

                            words[
                                del_info[count]["start_idx"]
                                + iex : del_info[count]["start_idx"]
                                + iex
                                + cur_item_len
                            ] = ["{}"] * cur_item_len

                        else:

                            words[del_info[count]["start_idx"] + iex]["word"] = (
                                ins_list[ins_inner_count]
                            )
                            ins_inner_count += 1

                else:
                    num_items_to_replace = (
                        del_info[count]["end_idx"] - del_info[count]["start_idx"]
                    )
                    words[del_info[count]["start_idx"] : del_info[count]["end_idx"]] = [
                        "{}"
                    ] * num_items_to_replace

                count += 1
        words = [w for w in words if w != "{}"]

        return words

    def compute_LLM_translation():
        # current_cost = cost_calculator.get_total_cost()

        t, tr, datas = ask_LLM_to_translate(texts)
        # print(f"Translation Cost: ${cost_calculator.get_total_cost() - current_cost}")
        # print(f"Total Cost: ${cost_calculator.get_total_cost()}")

        return {"datas": datas, "transcripts": t, "translates": tr}

    def compute_clean_translation(transcripts, translates):
        # current_cost = cost_calculator.get_total_cost()

        t, tr, datas = split_stage_one(transcripts, translates)
        # print(f"Cleaning Cost: ${cost_calculator.get_total_cost() - current_cost}")
        return {"transcripts": t, "translates": tr, "datas": datas}

    def compute_separated_sentences(transcripts, translates):
        # current_cost = cost_calculator.get_total_cost()
        # print("start separating")
        st, strl, datas = split_stage_two(transcripts, translates)
        # print(f"Separating Cost: ${cost_calculator.get_total_cost() - current_cost}")
        return {"seg_transcripts": st, "seg_translates": strl, "datas": datas}

    def compute_separated_long_tl(seg_transcripts, seg_translates):
        # print("start separating long tl")
        st, strl, datas = separate_too_long_tl(seg_transcripts, seg_translates)
        return {"seg_transcripts": st, "seg_translates": strl, "datas": datas}

    def compute_clean_sentences(seg_transcripts, seg_translates):
        # print("clean_sentences")
        st, strl = clean_sentences(seg_transcripts, seg_translates)
        return {"seg_transcripts": st, "seg_translates": strl}

    result = get_or_cache(f"{cache_path}/LLM_translation.toml", compute_LLM_translation)
    transcripts, translates = result["transcripts"], result["translates"]
    one_transcripts = copy.deepcopy(transcripts)
    words = check_words_equal(" ".join(texts), " ".join(one_transcripts), words)
    print(f'length check: {len(words)}/{len(" ".join(transcripts).split())}')

    # test = Redlines(" ".join(texts), " ".join(transcripts), markdown_style="none")
    # with open("redlines.md", "w", encoding="utf-8") as f:
    #     f.write(test.output_markdown)

    if len(transcripts) != len(translates):
        print("transcripts and translates not equal")
        print("LLM_translation.toml", len(transcripts), len(translates))
        sys.exit(1)
    # sys.exit(1)

    result = get_or_cache(
        f"{cache_path}/split_stage_one.toml",
        lambda: compute_clean_translation(transcripts, translates),
    )
    transcripts, translates = result["transcripts"], result["translates"]
    two_transcripts = copy.deepcopy(transcripts)

    words = check_words_equal(
        " ".join(one_transcripts), " ".join(two_transcripts), words
    )
    print(f'length check: {len(words)}/{len(" ".join(transcripts).split())}')

    # for idx, (a, b) in enumerate(zip(transcripts, translates)):
    #     if "。" in b[:-1]:
    #         print(f"[{idx}] \n{a}\n{b}\n")
    # sys.exit(1)

    if len(transcripts) != len(translates):
        print("transcripts and translates not equal")
        print("split_stage_one.toml", len(transcripts), len(translates))
        sys.exit(1)

    result = get_or_cache(
        f"{cache_path}/split_stage_two.toml",
        lambda: compute_separated_sentences(transcripts, translates),
    )
    seg_transcripts, seg_translates = (
        result["seg_transcripts"],
        result["seg_translates"],
    )
    if len(seg_transcripts) != len(seg_translates):
        print("seg_transcripts and seg_translates not equal")
        print("split_stage_two.toml", len(seg_transcripts), len(seg_translates))
        sys.exit(1)

    three_transcripts = copy.deepcopy(seg_transcripts)

    words = check_words_equal(
        " ".join(two_transcripts), " ".join(three_transcripts), words
    )
    print(f'length check: {len(words)}/{len(" ".join(seg_transcripts).split())}')

    result = get_or_cache(
        f"{cache_path}/split_stage_three.toml",
        lambda: compute_separated_long_tl(seg_transcripts, seg_translates),
    )
    seg_transcripts, seg_translates = (
        result["seg_transcripts"],
        result["seg_translates"],
    )
    if len(seg_transcripts) != len(seg_translates):
        print("seg_transcripts and seg_translates not equal")
        print("split_stage_three.toml", len(seg_transcripts), len(seg_translates))
        sys.exit(1)

    fourth_transcripts = copy.deepcopy(seg_transcripts)
    # with open("words.toml", "w", encoding="utf-8") as f:
    #     data = {
    #         "words": words,
    #         "a": " ".join(two_transcripts),
    #         "b": " ".join(three_transcripts),
    #     }
    #     toml.dump(data, f)
    words = check_words_equal(
        " ".join(three_transcripts),
        " ".join(fourth_transcripts),
        words,
    )
    print(f'length check: {len(words)}/{len(" ".join(seg_transcripts).split())}')

    result = get_or_cache(
        f"{cache_path}/clean_sentences.toml",
        lambda: compute_clean_sentences(seg_transcripts, seg_translates),
    )
    seg_transcripts, seg_translates = (
        result["seg_transcripts"],
        result["seg_translates"],
    )
    if len(seg_transcripts) != len(seg_translates):
        print("seg_transcripts and seg_translates not equal")
        print("clean_sentences.toml", len(seg_transcripts), len(seg_translates))
        sys.exit(1)

    fifth_transcripts = copy.deepcopy(seg_transcripts)
    words = check_words_equal(
        " ".join(fourth_transcripts),
        " ".join(fifth_transcripts),
        words,
    )
    print(f'length check: {len(words)}/{len(" ".join(seg_transcripts).split())}')

    # for idx, (a, b) in enumerate(zip(seg_transcripts, seg_translates)):
    #     char_ratio = round(len(re.findall(r"[\u4e00-\u9fff]", b)) / len(a.split()), 3)
    #     if char_ratio > 4:
    #         print(
    #             f"[{idx}] \n{a}\n{b}\n",
    #             char_ratio,
    #             "\n",
    #         )

    return seg_transcripts, seg_translates, words


def save_subtitle(subtitle_path, audio_path, result):
    writer = get_writer("all", subtitle_path)
    writer_args = {
        "highlight_words": False,
        "max_line_count": None,
        "max_line_width": None,
    }
    writer(result, audio_path, writer_args)


def get_deliver_video(
    cache_path, file_id, file_path, file_name, uploader, url, state=True, summary=[]
):
    title, output_video_path, thumbnail_path, tag, desc, season = load_video_info(
        cache_path, file_id, file_path, file_name, state
    )
    if not state:
        title = title.replace("【中字】", "")
    tid = load_channels()[uploader]
    # description需要根据其分区字符数限制进行调整
    if not os.path.exists("cache/delivery_videos.toml"):
        delivery_videos = []
    else:
        with open("cache/delivery_videos.toml", "rb") as toml_file:
            delivery_videos = tomllib.load(toml_file)["delivery_videos"]

    info_path = f"{cache_path}/{file_id}.toml"
    item = make_delivery_info(
        title, output_video_path, thumbnail_path, tag, desc, tid, url, season
    )
    item["MetaDataPath"] = info_path
    item["Summary"] = summary

    if item not in delivery_videos:
        delivery_videos.append(item)
        # with open(f"cache/delivery_videos.toml", "w") as toml_file:
        #     data = {"delivery_videos": delivery_videos}
        #     toml.dump(data, toml_file)
    return delivery_videos


comment_tasks = []


def deliver_and_save_completion(items, credential=None):
    lost_items = copy.deepcopy(items)
    for item in items:
        retry_count = 0
        max_retries = 20

        while retry_count < max_retries:
            try:
                res_ids = upload(
                    item["Video"],
                    item["Thumbnail"],
                    item["Tag"],
                    item["Description"],
                    item["Title"],
                    item["Season"],
                )
                info = load_cache(item["MetaDataPath"])
                info["aid"] = res_ids["aid"]
                info["bvid"] = res_ids["bvid"]
                info["summary"] = item["Summary"]
                save_cache(info, item["MetaDataPath"])
                # await deliver(item, credential)
                # 将id写入视频的meta data文件里，然后将当前时间，id，summary交给另一个线程去做置顶评论
                if item["Summary"]:
                    comment_tasks.append(
                        {
                            "time": datetime.now().timestamp(),
                            "bvid": res_ids["bvid"],
                            "aid": res_ids["aid"],
                            "summary": item["Summary"],
                        }
                    )

                break

            except Exception as e:
                save_cache(
                    {"delivery_videos": lost_items}, "cache/delivery_videos.toml"
                )
                raise e
                # with open(f"cache/delivery_videos.toml", "w") as toml_file:
                #     data = {"delivery_videos": lost_items}
                #     toml.dump(data, toml_file)

                if "601" in str(e):
                    # total_sleep_time = 1 * 3600  # 12小时转换为秒
                    current_time = datetime.now()
                    tomorrow_start = current_time.replace(
                        hour=0, minute=0, second=0, microsecond=0
                    ) + timedelta(days=1)
                    total_sleep_time = (tomorrow_start - current_time).total_seconds()
                    start_time = time.time()
                    while True:
                        elapsed_time = time.time() - start_time
                        if elapsed_time >= total_sleep_time:
                            break
                        hours, remaining_seconds = divmod(
                            int(total_sleep_time - elapsed_time), 3600
                        )
                        minutes, seconds = divmod(remaining_seconds, 60)
                        formatted_time = f"{hours:02}:{minutes:02}:{seconds:02}"

                        print(
                            f"Error code 601, temporary account risk control, waiting {formatted_time} and retry.",
                            end="\r",
                        )
                        time.sleep(1)  # 每秒更新一次
                    retry_count += 1
        save_completion(item)
        lost_items.remove(item)
    if os.path.exists("cache/delivery_videos.toml"):
        os.remove("cache/delivery_videos.toml")
    # print("finish delivering")


def get_cut_stamps(chunks):
    merged_chunks_period = merge_if_not_period(chunks)
    timestamps = find_closest_stamps(merged_chunks_period)

    return timestamps


def split_audio(timestamps, audio, clip_path):
    # print(timestamps)
    last_timestamp = 0.0
    audio = AudioSegment.from_file(audio)
    clip_audio_path = f"{clip_path}/audio"

    timestamps = [int(timestamp * 1000) for timestamp in timestamps]
    for i, timestamp in enumerate(timestamps):
        start_ms = last_timestamp
        end_ms = timestamp

        # 分割音频
        clip = audio[start_ms:end_ms]

        # 保存音频片段
        clip.export(f"{clip_audio_path}/{i + 1}.wav", format="wav")

        # 更新上一个时间戳
        last_timestamp = timestamp

    # 保存最后一段音频
    audio[last_timestamp:].export(
        f"{clip_audio_path}/{len(timestamps) + 1}.wav", format="wav"
    )


def split_video(timestamps, video_path, clip_path):
    # Load the video file
    video = VideoFileClip(video_path)
    last_timestamp = 0.0
    clip_video_path = f"{clip_path}/video"

    for i, timestamp in enumerate(timestamps):
        # Define start and end points in seconds
        start = last_timestamp
        end = timestamp

        # Extract the clip
        clip = video.subclip(start, end)

        # Save the clip
        clip.write_videofile(f"{clip_video_path}/{i + 1}.mp4", codec="libx264")

        # Update the last timestamp
        last_timestamp = timestamp

    # Save the last clip, from the last timestamp to the end of the video
    video.subclip(last_timestamp, video.duration).write_videofile(
        f"{clip_video_path}/{len(timestamps) + 1}.mp4", codec="libx264"
    )


def process_task(info, key, save_path, task_function, *args):
    """
    根据字典中的键值判断是否执行特定任务，并在执行后更新字典。

    :param info: 字典，用于存储任务执行状态
    :param key: 字典中用于判断任务是否已完成的键
    :param save_path: 保存更新后字典的路径
    :param task_function: 需要执行的任务函数
    :param args: 传递给任务函数的参数
    """
    if not info.get(key, False):
        try:
            task_function(*args)  # 执行任务函数
            info[key] = True  # 更新任务状态
            save_cache(info, save_path)  # 保存更新后的字典
        except Exception as e:
            raise e


def waiting_thread(threads):
    if threads:
        for thread in threads:
            flag = [True]
            print_status(flag, "Waiting for thread to finish...")
            thread.join()
            flag[0] = False
        threads.clear()


def add_addition_videos(additions, videos):
    if additions:
        # 将 videos 转换为集合以提高查找效率
        # videos = set(video for video in videos)
        # print(additions)
        urls_in_additions = [addition.get("url") for addition in additions]

        add_item = [item for item in videos if item.get("url") in urls_in_additions]
        if not add_item:
            add_item = additions
        # print(add_item)
        filtered_set = [
            item for item in videos if item.get("url") not in urls_in_additions
        ]
        final_videos = add_item + filtered_set
    else:
        final_videos = videos

    return final_videos


def filter_channel(channel, videos):
    if channel:
        videos = [item for item in videos if item.get("uploader") in channel]

    return videos
