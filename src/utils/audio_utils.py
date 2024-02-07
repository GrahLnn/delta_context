import os, torch, librosa, soundfile as sf, numpy as np, shutil
import subprocess

# from audio_separator import Separator
from pyannote.audio import Pipeline
from mdx23.inference import start
from src.env.env import HUGGINGFACE_TOKEN
import sys
from voicefixer import VoiceFixer
from .status_utils import print_status

# def vad_clean(audio_path, output_path, output_name):
#     # Load the Silero VAD model
#     # Make sure torch is installed in your environment
#     device = torch.device(
#         "cuda" if torch.cuda.is_available() else "cpu"
#     )  # or 'cuda' if available
#     model, utils = torch.hub.load(
#         repo_or_dir="snakers4/silero-vad",
#         model="silero_vad",
#         # force_reload=True,
#         trust_repo=True,
#     )  # Added trust_repo=True to avoid the warning

#     (get_speech_timestamps, _, read_audio, _, _) = utils

#     # Load your audio file with librosa
#     audio, sr = librosa.load(audio_path, sr=16000)
#     audio = audio.astype(np.float32)

#     # Get speech timestamps
#     speech_timestamps = get_speech_timestamps(audio, model, threshold=0.3)

#     # Mute non-speech parts based on timestamps
#     muted_audio = np.zeros_like(audio)
#     for timestamp in speech_timestamps:
#         start_sample = timestamp["start"]
#         end_sample = timestamp["end"]
#         muted_audio[start_sample:end_sample] = audio[start_sample:end_sample]


#     # Save the processed audio file
#     sf.write(f"{output_path}/{output_name}.wav", muted_audio, sr)


def calculate_energy(audio):
    """计算音频的能量"""
    return np.sum(audio**2) / len(audio)


def vad_clean(audio_name, output_path, name):
    voicefixer = VoiceFixer()
    enhance_folder = f"{output_path}/enhance_speaker"
    folder = f"{output_path}/clean_speaker"
    fixer_audio = f"{folder}/clean_fixer.wav"
    if not os.path.exists(folder):
        os.makedirs(folder)
    if not os.path.exists(enhance_folder):
        os.makedirs(enhance_folder)

    shutil.copy(audio_name, f"{enhance_folder}/{name}.wav")
    # command = ["resemble-enhance", enhance_folder, enhance_folder, "--denoise_only"]

    # 调用命令
    # subprocess.run(command, capture_output=True, text=True)

    # print("voicefixer start")
    voicefixer.restore(
        input=f"{enhance_folder}/{name}.wav", output=fixer_audio, cuda=True, mode=0
    )

    # 加载预训练的模型
    print("speaker diarization start")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HUGGINGFACE_TOKEN,
    )

    # 将管道发送到 GPU（如果可用）
    pipeline.to(torch.device("cuda"))

    # 应用预训练的管道
    diarization = pipeline(fixer_audio)
    print("speaker diarization finish")
    # 读取原始音频文件
    audio, sr = librosa.load(fixer_audio, sr=None)
    os.remove(f"{folder}/clean_fixer.wav")

    # 创建一个字典来存储每个说话人的音频段
    speakers_audio = {}

    # 设置能量阈值
    energy_threshold = 0.0005  # 根据您的需求调整

    # 遍历分割结果，提取每个说话人的音频
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        # 如果之前没有处理过这个说话人，初始化一个空列表
        if speaker not in speakers_audio:
            speakers_audio[speaker] = []

        # 提取当前说话人的音频段及其开始和结束时间
        start_sample = int(turn.start * sr)
        end_sample = int(turn.end * sr)
        speaker_segment = {
            "audio": audio[start_sample:end_sample],
            "start": turn.start,
            "end": turn.end,
        }
        speakers_audio[speaker].append(speaker_segment)
        print(
            f"Speaker {speaker}'s audio energy: {calculate_energy(speaker_segment['audio']):.6f}"
        )

    final_audio = np.zeros_like(audio)
    # 将每个说话人的音频段连接起来并保存为文件
    for speaker, segments in speakers_audio.items():
        # 连接所有段落
        speaker_audio = np.concatenate([audio_seg["audio"] for audio_seg in segments])

        # 计算连接后音频的能量
        audio_energy = calculate_energy(speaker_audio)
        print(f"Speaker {speaker}'s audio energy: {audio_energy:.6f}")
        del speaker_audio

        # # 计算所有音频段开始时间的标准差
        # start_times = [segment["start"] for segment in segments]
        # start_times_std = np.std(start_times)
        # # 标准化标准差
        # total_duration = len(audio) / sr
        # normalized_std = start_times_std / total_duration
        # print(f"Normalized Start Times Standard Deviation: {normalized_std:.2f}")

        # 计算总覆盖时长
        total_coverage = sum(
            [segment["end"] - segment["start"] for segment in segments]
        )
        # 计算整个音频的时长
        total_duration = len(audio) / sr

        # 计算覆盖比例
        coverage_ratio = total_coverage / total_duration
        print(f"Coverage Ratio: {coverage_ratio:.2f}")
        # 判断能量是否高于阈值
        if audio_energy > energy_threshold:
            # 创建一个与原始音频等长的静默音频数组
            # final_audio = np.zeros_like(audio)

            # 对于每个说话人的音频段，将其填充到最终音频中
            for segment in segments:
                start_sample = int(segment["start"] * sr)
                end_sample = int(segment["end"] * sr)
                final_audio[start_sample:end_sample] = segment["audio"]
        print(
            f"Speaker {speaker}'s final audio energy: {calculate_energy(final_audio):.6f}"
        )

        #     # 保存音频文件
        #     output_name = f"{folder}/{speaker}.wav"
        #     sf.write(output_name, final_audio, sr)
        # if not (audio_energy > energy_threshold and coverage_ratio > 0.1):
        # if not (audio_energy > energy_threshold):
        #     # 静默不符合条件的音频段
        #     for segment in segments:
        #         start_sample = int(segment["start"] * sr)
        #         end_sample = int(segment["end"] * sr)
        #         audio[start_sample:end_sample] = 0
    # del audio
    output_name = f"{folder}/processed_audio.wav"
    sf.write(output_name, final_audio, sr)
    # sys.exit()
    speakers = os.listdir(folder)
    if len(speakers) == 1:
        path_speakers = [f"{folder}/{speaker}" for speaker in speakers]
        shutil.copy(path_speakers[0], f"{output_path}/{name}.wav")
    elif len(speakers) > 1:
        print(f"{audio_name} 多个说话人，该怎么办呢...怎么办呢...")
        sys.exit()
    else:
        final_audio = np.zeros_like(audio)
        output_name = f"{output_path}/{name}.wav"
        sf.write(output_name, final_audio, sr)


def extract_vocal(audio_path, output_path, output_name):
    if not os.path.exists(f"{output_path}/clean_vocal"):
        os.makedirs(f"{output_path}/clean_vocal")
    arg = {
        "input_audio": [
            f"{audio_path}",
        ],
        "output_folder": f"{output_path}/clean_vocal",
        "cpu": False,
        "overlap_demucs": 0.1,
        "overlap_VOCFT": 0.1,
        "overlap_VitLarge": 1,
        "overlap_InstVoc": 1,
        "weight_InstVoc": 8,
        "weight_VOCFT": 1,
        "weight_VitLarge": 5,
        "single_onnx": False,
        "large_gpu": True,
        "BigShifts": 7,
        "vocals_only": True,
        "use_VOCFT": False,
        "output_format": "FLOAT",
    }
    start(arg)

    flag = [True]
    # print_status(flag, "vad clean")
    vad_clean(
        f"{output_path}/clean_vocal/{output_name}_vocals.wav",
        output_path,
        output_name,
    )
    flag[0] = False
    # shutil.move(
    #     f"{output_path}/clean_vocal/{output_name}.wav",
    #     audio_path,
    # )
    # sys.exit()
    # print("finish vad clean")
    # os.rename(
    #     f"{output_path}/{output_name}_vocals.wav",
    #     f"{output_path}/{output_name}.wav",
    # )
    # separator = Separator(
    #     audio_path,
    #     model_name="UVR-MDX-NET-Inst_HQ_1",
    #     secondary_stem_path=audio_path,
    #     # use_cuda=True,
    #     output_single_stem="vocals",
    # )

    # # Perform the separation
    # separator.separate()
