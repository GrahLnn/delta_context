import whisperx, torch, os, sys, re, string
from pydub import AudioSegment


from src.env.env import DEVICE, COMPUTE_TYPE
from src.utils.status_utils import print_status
from src.utils.cache_utils import load_cache, save_cache
from src.utils.LLM_utils import get_completion


def recursive_float_conversion(data, keys_to_convert):  # 不该出现在此
    if isinstance(data, dict):
        for key, value in data.items():
            if key in keys_to_convert and isinstance(value, str):
                try:
                    data[key] = float(value)
                except ValueError:
                    pass
            elif isinstance(value, (dict, list)):
                recursive_float_conversion(value, keys_to_convert)
    elif isinstance(data, list):
        for item in data:
            recursive_float_conversion(item, keys_to_convert)
    return data


def align_transcripts(seg_transcripts, audio_path, cache_path):
    if os.path.exists(f"{cache_path}/align_result.toml"):
        result = load_cache(f"{cache_path}/align_result.toml")
        keys_to_convert = ["start", "end", "score"]
        result = recursive_float_conversion(result, keys_to_convert)
        ### test
        words = result["word_segments"]
        for item in words:
            if not item.get("start", ""):
                print(item)
    else:
        flag = [True]
        try:
            print_status(flag, desc="Align with transcript")
            # seg_transcripts = [
            #     item.strip().strip(string.punctuation) + "." for item in seg_transcripts
            # ]
            composite_transcripts = " ".join(list(seg_transcripts))

            result = {"segments": []}

            audio = AudioSegment.from_file(audio_path)

            duration_in_seconds = round(len(audio) / 1000.0, 3)

            new_segment = {
                "text": composite_transcripts.strip().replace("  ", " "),
                "start": 0,
                "end": duration_in_seconds,
            }
            result["segments"].append(new_segment)
            audio = whisperx.load_audio(audio_path)
            model_a, metadata = whisperx.load_align_model(
                language_code="en", device=DEVICE
            )
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                DEVICE,
                return_char_alignments=False,
            )
            import gc

            gc.collect()
            torch.cuda.empty_cache()
            del model_a
            # print(len(result["segments"]), len(seg_translates))
            # for idx, item in enumerate(seg_translates):
            #     result["segments"][idx]["translation"] = item

            flag[0] = False

            save_cache(result, f"{cache_path}/align_result.toml")
        except Exception as e:
            print(e)
            flag[0] = False
            sys.exit()
    return result
