import os, sys, torch, re, copy
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from pydub import AudioSegment
import librosa
from asset.env.env import DEVICE, BATCH_SIZE, COMPUTE_TYPE, LIMIT_LEN_TIME
from .status_utils import print_status, _CustomProgressBar
from .LLM_utils import get_completion
from .transcribe_auxiliary_utils import split_on_second_last_dot
from tqdm import tqdm
from whisper.tokenizer import get_tokenizer
from transformers import GenerationConfig
from .whisper_analysis.dtw_analysis_hf import get_words_timestamp
import whisper
import subprocess
import shutil
import string


def transcribe_with_ins_whisper(audio_file, timestamps=False):

    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
        torch_dtype=torch.float16,
        device="cuda",  # or mps for Mac devices
        model_kwargs={"use_flash_attention_2": False},
    )

    outputs = pipe(
        audio_file,
        chunk_length_s=30,
        batch_size=1,
        return_timestamps="word",
        generate_kwargs={"task": "transcribe", "language": None},
    )
    outputs["text"] = outputs["text"].strip()
    # with open(f"{save_path}/transcribe_outputs.toml", "w", encoding="utf-8") as f:
    #     toml.dump(outputs, f)


def transcribe_with_whisper(audio_file, timestamps=False):
    # start_time = datetime.now()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "distil-whisper/distil-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=25,
        batch_size=16,
        torch_dtype=torch_dtype,
        device=device,
    )

    # src_audio = "video/1_MapKit_with_SwiftUI__Regions_Markers_and_Annotations_and_CameraPosition/9xzHJAT_Iqk.mp3"

    split_audio(audio_file, "test_output")

    files = os.listdir("test_output")
    files.sort(key=natural_keys)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).cuda()
    processor = WhisperProcessor.from_pretrained(model_id)
    answer = []
    for i, file in enumerate(tqdm(files, desc="Transcribing")):
        # print(file)
        resulta = pipe(f"test_output/{file}")
        # print(resulta["text"])
        clip_audio, _ = librosa.load(f"test_output/{file}", sr=16000)
        stamps = get_words_timestamp(model, processor, clip_audio, resulta["text"])
        for ii in range(len(stamps)):
            stamps[ii][1] += i * 25
            stamps[ii][1] = round(stamps[ii][1], 2)
            stamps[ii][2] += i * 25
            stamps[ii][2] = round(stamps[ii][2], 2)
        if answer and stamps[0][0].strip() == answer[-1]["word"].strip():
            answer[-1]["end"] = stamps[0][2]
            stamps = stamps[1:]
        stamps = [
            {"word": str(n[0]), "start": str(n[1]), "end": str(n[2])} for n in stamps
        ]
        answer.extend(stamps)
        # rm = "".join([text[0] for text in stamps])
        # # transcription = transcription.replace(rm, "")
        # print(stamps)
        # print(rm)

    # print(len(answer))
    shutil.rmtree("test_output")
    answer = clean(answer)
    text = "".join([text["word"] for text in answer]).strip()
    new_words = []
    for word in answer:
        if word["word"].startswith(" "):
            new_words.append(word)
        else:
            try:
                new_words[-1]["word"] += word["word"]
                new_words[-1]["end"] = word["end"]
            except IndexError:
                new_words.append(word)
    return {"text": text, "words": new_words}


def transcribe_with_openai_whisper(audio_file, timestamps=False):
    flag = [True]
    desc = [""]
    print_status(flag, desc=desc)
    transcribe_module = sys.modules["whisper.transcribe"]
    transcribe_module.tqdm.tqdm = _CustomProgressBar
    try:
        desc[0] = "load model"
        model = whisper.load_model("large-v3")
        flag[0] = False
        # model = whisper.load_model("distil-whisper/distil-large-v2")
        # print("transcribe start")

        result = model.transcribe(
            audio_file,
            # suppress_tokens=[-1] + number_tokens,
            word_timestamps=True,
            compression_ratio_threshold=1.5,
            fp16=False,
            verbose=None,
            language="en",
            # temperature=0,
        )

        new_result = {"text": "", "chunks": [], "language": ""}
        new_result["text"] = result["text"].strip()
        new_result["language"] = result["language"]
        new_result["chunks"] = result["segments"]

        # if new_result["language"] != "en":
        #     raise ValueError(f"Language is not English, {new_result['language']}")
        for idx, chunk in enumerate(new_result["chunks"]):
            new_result["chunks"][idx]["timestamp"] = [
                round(float(chunk["start"]), 2),
                round(float(chunk["end"]), 2),
            ]
            for word in chunk["words"]:
                word["start"] = str(round(float(word["start"]), 2))
                word["end"] = str(round(float(word["end"]), 2))

            new_words = []
            for word in chunk["words"]:
                if word["word"].startswith(" "):
                    new_words.append(word)
                else:
                    try:
                        new_words[-1]["word"] += word["word"]
                        new_words[-1]["end"] = word["end"]
                    except IndexError:
                        new_words.append(word)
            new_result["chunks"][idx]["words"] = new_words

        return new_result
    except Exception as e:
        desc[0] = "Transcribe failed"
        flag[0] = False
        raise e


# def transcribe_with_distil_whisper(audio_file, timestamps=False):
#     # tokenizer = get_tokenizer(
#     #     multilingual=True
#     # )  # 使用 multilingual=True 如果使用多语言模型
#     # number_tokens = [
#     #     i
#     #     for i in range(tokenizer.eot)
#     #     if all(c in "0123456789" for c in tokenizer.decode([i]).removeprefix(" "))
#     # ]
#     flag = [True]
#     desc = [""]
#     print_status(flag, desc=desc)
#     transcribe_module = sys.modules['whisper.transcribe']
#     transcribe_module.tqdm.tqdm = _CustomProgressBar
#     try:


#         # device = "cuda:0" if torch.cuda.is_available() else "cpu"
#         # torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

#         # model_id = "distil-whisper/distil-large-v2"
#         # # model_id = "openai/whisper-large-v3"
#         # generation_config = GenerationConfig.from_pretrained(model_id)
#         # generation_config.suppress_tokens += number_tokens

#         # model = AutoModelForSpeechSeq2Seq.from_pretrained(
#         #     model_id,
#         #     torch_dtype=torch_dtype,
#         #     low_cpu_mem_usage=True,
#         #     use_safetensors=True,
#         #     # use_flash_attention_2=True,
#         # )
#         # # model = model.to_bettertransformer()
#         # model.to(device)

#         # processor = AutoProcessor.from_pretrained(model_id)

#         # pipe = pipeline(
#         #     "automatic-speech-recognition",
#         #     model=model,
#         #     tokenizer=processor.tokenizer,
#         #     feature_extractor=processor.feature_extractor,
#         #     max_new_tokens=128,
#         #     chunk_length_s=15,
#         #     batch_size=16,
#         #     # batch_size=1,
#         #     torch_dtype=torch_dtype,
#         #     device=device,
#         #     generate_kwargs={"task": "transcribe", "language": None},
#         # )

#         # result = pipe(
#         #     audio_file,
#         #     return_timestamps=timestamps,
#         #     suppress_tokens=[-1] + number_tokens,
#         # )

#         # pipe = pipeline(
#         #     "automatic-speech-recognition",
#         #     model="distil-whisper/distil-large-v2",
#         #     torch_dtype=torch.float16,
#         #     device="cuda",  # or mps for Mac devices
#         #     model_kwargs={"use_flash_attention_2": False},
#         # )

#         # result = pipe(
#         #     audio_file,
#         #     chunk_length_s=30,
#         #     batch_size=1,
#         #     return_timestamps="word",
#         #     generate_kwargs={"task": "transcribe", "language": None},
#         # )
#         # tokenizer = get_tokenizer(
#         #     multilingual=False
#         # )  # use multilingual=True if using multilingual model
#         # number_tokens = [
#         #     i
#         #     for i in range(tokenizer.eot)
#         #     if all(c in "0123456789" for c in tokenizer.decode([i]).removeprefix(" "))
#         # ]
#         # print("load model")


#         desc[0] = "load model"
#         model = whisper.load_model("large-v2")
#         flag[0] = False
#         # model = whisper.load_model("distil-whisper/distil-large-v2")
#         # print("transcribe start")

#         result = model.transcribe(
#             audio_file,
#             # suppress_tokens=[-1] + number_tokens,
#             word_timestamps=True,
#             compression_ratio_threshold=1.5,
#             fp16=False, verbose=None
#             # temperature=0,
#         )


#         # print(result)
#         new_result = {"text": "", "chunks": [], "language": ""}
#         new_result["text"] = result["text"].strip()
#         new_result["language"] = result["language"]
#         new_result["chunks"] = result["segments"]

#         if new_result["language"] != "en":
#             return new_result
#         for idx, chunk in enumerate(new_result["chunks"]):
#             new_result["chunks"][idx]["timestamp"] = [
#                 round(float(chunk["start"]), 2),
#                 round(float(chunk["end"]), 2),
#             ]
#             for word in chunk["words"]:
#                 word["start"] = str(round(float(word["start"]), 2))
#                 word["end"] = str(round(float(word["end"]), 2))

#             new_words = []
#             for word in chunk["words"]:
#                 if word["word"].startswith(" "):
#                     new_words.append(word)
#                 else:
#                     new_words[-1]["word"] += word["word"]
#                     new_words[-1]["end"] = word["end"]
#             new_result["chunks"][idx]["words"] = new_words

#         # for idx, chunk in enumerate(result["chunks"]):
#         #     result["chunks"][idx]["timestamp"] = list(chunk["timestamp"])

#         # if None in result["chunks"][-1]["timestamp"]:
#         #     # 计算音频总长度然后赋值给最后一个时间戳
#         #     audio = AudioSegment.from_file(audio_file)
#         #     audio_length = audio.duration_seconds
#         #     result["chunks"][-1]["timestamp"][1] = audio_length


#         return new_result
#     except Exception as e:
#         desc[0] = "Transcribe failed"
#         flag[0] = False
#         raise e


def first_proofread(sentences):
    prompt = "Proofread follow speaker transcription:"

    for idx, segment in enumerate(tqdm(sentences, desc="First Proofreading")):
        message = f"{prompt}\n```\n{segment}\n```"
        sentences[idx] = get_completion(message).strip('"').strip("```").strip()

    return sentences


def proofread(video_info, sentences):
    # print("Start Second Proofreading")
    # summary = get_completion(f"summary in one sentence:\n{video_info}")
    # term = get_completion(f"extract all the terms:\n{video_info}")

    # term_explain = get_completion(
    #     f"According to this template.\nBQN (pronunciation like Bee-Quen), APL (pronunciation like Ayy-Pee-Ell), , Uiua (pronunciation like wee-wuh)\nFinishing the following words:\n{term}",
    #     model="gpt-4",
    # )
    # 这里应该直接去看term然后替换

    #     for idx, segment in enumerate(tqdm(sentences, desc="Proofreading")):
    #         prompt = f"""Based this context: {summary}

    # {term_explain}

    # Now proofread follow speaker transcription especially words that appear out of nowhere.:
    # "{segment}"
    # """
    #         # print(prompt)
    #         sentences[idx] = (
    #             get_completion(prompt).strip('"').strip("```").strip("'").strip()
    #         )
    # print(sentences[idx])
    return sentences


def shifting_sentence(texts):
    modified_texts = split_on_second_last_dot(texts)
    return modified_texts


# 合并具有相同头尾时间戳的文本
def merge_chunks_for_check(chunks):
    merged_chunks = []
    i = 0
    while i < len(chunks):
        current_chunk = copy.deepcopy(chunks[i])
        # 检查是否有下一个元素且当前元素的结束时间戳与下一个元素的开始时间戳相同
        while (
            i + 1 < len(chunks)
            and current_chunk["timestamp"][1] == chunks[i + 1]["timestamp"][0]
        ):
            # 合并文本并更新结束时间戳
            current_chunk["text"] += " " + chunks[i + 1]["text"].strip()
            current_chunk["timestamp"][1] = chunks[i + 1]["timestamp"][1]
            current_chunk["words"] += chunks[i + 1]["words"]
            i += 1
        merged_chunks.append(current_chunk)
        i += 1
    return merged_chunks


def merge_chunks(stamps, whisper_chunks):
    merged_chunks = []
    stamps = iter(stamps)
    end_point = next(stamps)
    cur_chunk = ""
    cur_words = []
    for chunk in whisper_chunks:
        if end_point and chunk["timestamp"][1] == end_point:
            cur_chunk += chunk["text"]
            cur_words += chunk["words"]
            merged_chunks.append({"text": cur_chunk, "words": cur_words})
            cur_chunk = ""
            cur_words = []

            try:
                end_point = next(stamps)
            except StopIteration:
                end_point = None
        else:
            cur_chunk += chunk["text"]
            cur_words += chunk["words"]
    merged_chunks.append({"text": cur_chunk, "words": cur_words})
    return merged_chunks


# 检测文本是否以句号结尾的函数
# 如果不是，则与下一个文本项连接，然后再次检测
def merge_if_not_period(chunks):
    merged_chunks = []
    i = 0
    while i < len(chunks):
        current_chunk = chunks[i]
        # 检查当前文本是否以句号结尾
        while i + 1 < len(chunks) and not current_chunk["text"].strip().endswith("."):
            # 不以句号结尾，与下一个文本合并

            current_chunk["text"] += " " + chunks[i + 1]["text"].strip()
            current_chunk["timestamp"][1] = chunks[i + 1]["timestamp"][1]
            current_chunk["words"] += chunks[i + 1]["words"]
            i += 1
        merged_chunks.append(current_chunk)
        i += 1
    return merged_chunks


def concatenate_sentences_punc_end(sentences):
    processed_list = []
    for item in sentences:
        if processed_list and processed_list[-1][-1] not in (
            ".",
            "?",
            "!",
            ";",
            ":",
        ):
            processed_list[-1] += " " + item

        else:
            processed_list.append(item)

    return processed_list


def concatenate_sentences_en_end(sentences):
    processed_list = []
    for item in sentences:
        if processed_list and re.match(r"[a-zA-Z]$", processed_list[-1][-1]):
            processed_list[-1] += "，" + item

        else:
            processed_list.append(item)

    return processed_list


def find_closest_stamps(merged_chunks):
    interval = LIMIT_LEN_TIME * 60
    closest_stamps = []

    # 初始化目标时间
    target_time = interval
    end_time = [chunk["timestamp"][1] for chunk in merged_chunks]

    for idx, time in enumerate(end_time):
        if time > target_time:
            closest_stamp = end_time[idx - 1]
            closest_stamps.append(closest_stamp)

            target_time = interval + closest_stamp

    return closest_stamps


def get_audio_length_ffmpeg(audio_path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    duration = float(result.stdout)
    return duration


def split_audio(input_path, output_folder, segment_length=25, output_format="wav"):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    audio_length = get_audio_length_ffmpeg(input_path)

    num_segments = int(audio_length // segment_length) + (
        1 if audio_length % segment_length else 0
    )

    for i in tqdm(range(num_segments), desc="Splitting Audio"):
        start_time = i * segment_length
        duration = min(segment_length, audio_length - start_time)
        output_path = os.path.join(
            output_folder,
            f"segment_{i + 1}.{output_format}",
        )
        if os.path.exists(output_path):
            continue
        subprocess.run(
            [
                "ffmpeg",
                "-ss",
                str(start_time),
                "-i",
                input_path,
                "-t",
                str(duration),
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                output_path,
            ],
            stdout=subprocess.DEVNULL,  # 抑制标准输出
            stderr=subprocess.DEVNULL,  # 抑制错误输出
        )


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) 将会以人类的方式对列表进行排序
    """
    return [atoi(c) for c in re.split(r"(\d+)", text)]


def clean(words: list[dict]):
    new_words = []
    for idx, word in enumerate(words):
        new_word = word.copy()
        new_word["idx"] = idx
        new_words.append(new_word)

    bad_sequence = []
    cur_words = []
    for word in new_words:
        word["word"] = word["word"].strip().strip(string.punctuation).lower()
        if not cur_words:
            cur_words.append(word)
        elif float(word["start"]) - float(cur_words[-1]["end"]) < 5:
            cur_words.append(word)
        else:
            bad_sequence.append(cur_words)
            cur_words = [word]

    vaild_check = []
    surface_len = 40
    for idx, sequence in enumerate(bad_sequence):
        words_seq = set([word["word"] for word in sequence])
        if len(list(words_seq)) / len(sequence) < 0.8 and len(sequence) < surface_len:

            vaild_check.append(False)
        elif (
            len(sequence) < surface_len
            and idx > 0
            and len(
                list(
                    set(
                        [word["word"] for word in sequence]
                        + [word["word"] for word in bad_sequence[idx - 1]]
                    )
                )
            )
            / (len(sequence) + len(bad_sequence[idx - 1]))
            < len(list(set([word["word"] for word in bad_sequence[idx - 1]])))
            / (len(bad_sequence[idx - 1]))
        ):
            vaild_check.append(False)
        else:
            vaild_check.append(True)

    for idx, check in enumerate(vaild_check):
        if check:
            if idx > 0 and idx < len(vaild_check) - 1:
                if not vaild_check[idx - 1] and not vaild_check[idx + 1]:
                    vaild_check[idx] = False

    for idx, check in enumerate(vaild_check):
        if len(bad_sequence[idx]) < 50:
            if len([word["word"] for word in bad_sequence[idx]]) < 5:
                vaild_check[idx] = False
            # print(
            #     idx,
            #     vaild_check[idx],
            #     len([word["word"] for word in bad_sequence[idx]]),
            #     [word["word"] for word in bad_sequence[idx]],
            # )
            # print()

    start = 0
    end = 0
    for idx, check in enumerate(vaild_check):
        if len(bad_sequence[idx]) < 50:
            if not check and idx - end <= 1:
                end = idx
            else:
                break

    if start == end:
        return words

    remove_list = []
    for idx in range(start, end + 1):
        remove_list += bad_sequence[idx]

    remove_list = [word["idx"] for word in remove_list]

    remove_list = sorted(remove_list, reverse=True)
    # print(remove_list)

    for idx in remove_list:
        words.pop(idx)

    x_words = []
    for idx, word in enumerate(words):
        if not x_words:
            x_words.append(word)
        elif word != x_words[-1]:
            x_words.append(word)

    return x_words
