import os, re, json, sys
from typing import Callable, Optional, TextIO


from src.env.env import ADD_TRANSITION, target_language, ASS_STYLE

languages_dirc = {
    "Chinese": "zh",
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
}
lang = languages_dirc[target_language]


def format_timestamp(
    seconds: float, always_include_hours: bool = False, decimal_marker: str = "."
):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )


class ResultWriter:
    extension: str

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def __call__(self, result: dict, audio_path: str, options: dict):
        audio_basename = os.path.basename(audio_path)
        audio_basename = os.path.splitext(audio_basename)[0]
        output_path = os.path.join(
            self.output_dir, audio_basename + "." + self.extension
        )

        with open(output_path, "w", encoding="utf-8") as f:
            self.write_result(result, file=f, options=options)

    def write_result(self, result: dict, file: TextIO, options: dict):
        raise NotImplementedError


class WriteTXT(ResultWriter):
    extension: str = "txt"

    def write_result(self, result: dict, file: TextIO, options: dict):
        for segment in result["segments"]:
            print(segment["text"].strip(), file=file, flush=True)


class SubtitlesWriter(ResultWriter):
    always_include_hours: bool
    decimal_marker: str

    def iterate_result(self, result: dict, options: dict):
        raw_max_line_width: Optional[int] = options["max_line_width"]
        max_line_count: Optional[int] = options["max_line_count"]
        highlight_words: bool = options["highlight_words"]
        max_line_width = 1000 if raw_max_line_width is None else raw_max_line_width
        preserve_segments = max_line_count is None or raw_max_line_width is None

        if len(result["segments"]) == 0:
            return

        def iterate_subtitles():
            line_len = 0
            line_count = 1
            # the next subtitle to yield (a list of word timings with whitespace)
            subtitle: list[dict] = []
            times = []
            last = result["segments"][0]["start"]
            for segment in result["segments"]:
                for i, original_timing in enumerate(segment["words"]):
                    timing = original_timing.copy()
                    long_pause = not preserve_segments
                    if "start" in timing:
                        long_pause = long_pause and timing["start"] - last > 3.0
                    else:
                        long_pause = False
                    has_room = line_len + len(timing["word"]) <= max_line_width
                    seg_break = i == 0 and len(subtitle) > 0 and preserve_segments
                    if line_len > 0 and has_room and not long_pause and not seg_break:
                        # line continuation
                        line_len += len(timing["word"])
                    else:
                        # new line
                        timing["word"] = timing["word"].strip()
                        if (
                            len(subtitle) > 0
                            and max_line_count is not None
                            and (long_pause or line_count >= max_line_count)
                            or seg_break
                        ):
                            # subtitle break
                            yield subtitle, times
                            subtitle = []
                            times = []
                            line_count = 1
                        elif line_len > 0:
                            # line break
                            line_count += 1
                            timing["word"] = "\n" + timing["word"]
                        line_len = len(timing["word"].strip())
                    subtitle.append(timing)
                    times.append(
                        (segment["start"], segment["end"], segment.get("speaker"))
                    )
                    if "start" in timing:
                        last = timing["start"]
            if len(subtitle) > 0:
                yield subtitle, times

        if "words" in result["segments"][0]:
            for subtitle, _ in iterate_subtitles():
                sstart, ssend, speaker = _[0]
                subtitle_start = self.format_timestamp(sstart)
                subtitle_end = self.format_timestamp(ssend)
                subtitle_text = " ".join([word["word"] for word in subtitle])
                has_timing = any(["start" in word for word in subtitle])

                # add [$SPEAKER_ID]: to each subtitle if speaker is available
                prefix = ""
                if speaker is not None:
                    prefix = f"[{speaker}]: "

                if highlight_words and has_timing:
                    last = subtitle_start
                    all_words = [timing["word"] for timing in subtitle]
                    for i, this_word in enumerate(subtitle):
                        if "start" in this_word:
                            start = self.format_timestamp(this_word["start"])
                            end = self.format_timestamp(this_word["end"])
                            if last != start:
                                yield last, start, subtitle_text

                            yield start, end, prefix + " ".join(
                                [
                                    re.sub(r"^(\s*)(.*)$", r"\1<u>\2</u>", word)
                                    if j == i
                                    else word
                                    for j, word in enumerate(all_words)
                                ]
                            )
                            last = end
                else:
                    yield subtitle_start, subtitle_end, prefix + subtitle_text
        else:
            for segment in result["segments"]:
                segment_start = self.format_timestamp(segment["start"])
                segment_end = self.format_timestamp(segment["end"])
                segment_text = segment["text"].strip().replace("-->", "->")
                if "speaker" in segment:
                    segment_text = f"[{segment['speaker']}]: {segment_text}"
                yield segment_start, segment_end, segment_text

    def format_timestamp(self, seconds: float):
        return format_timestamp(
            seconds=seconds,
            always_include_hours=self.always_include_hours,
            decimal_marker=self.decimal_marker,
        )


class WriteSRT4T(SubtitlesWriter):
    extension: str = "srt"
    always_include_hours: bool = True
    decimal_marker: str = ","

    def __call__(self, result: dict, audio_path: str, options: dict):
        audio_basename = os.path.basename(audio_path)
        audio_basename = os.path.splitext(audio_basename)[0]
        output_path = os.path.join(
            self.output_dir, f"{audio_basename}_{lang}.{self.extension}"
        )  # 在这里修改输出文件名

        with open(output_path, "w", encoding="utf-8") as f:
            self.write_result(result, file=f, options=options)

    def write_result(self, result: dict, file: TextIO, options: dict):
        for i, (start, end, text) in enumerate(
            self.iterate_translation(result, options), start=1
        ):
            print(f"{i}\n{start} --> {end}\n{text}\n", file=file, flush=True)

    def iterate_translation(self, result: dict, options: dict):
        for segment in result["segments"]:
            segment_start = self.format_timestamp(segment["start"])
            segment_end = self.format_timestamp(segment["end"])
            segment_translation = (
                segment.get("translation", "").strip().replace("-->", "->")
            )
            yield segment_start, segment_end, segment_translation


class WriteVTT(SubtitlesWriter):
    extension: str = "vtt"
    always_include_hours: bool = False
    decimal_marker: str = "."

    def write_result(self, result: dict, file: TextIO, options: dict):
        print("WEBVTT\n", file=file)
        for start, end, text in self.iterate_result(result, options):
            print(f"{start} --> {end}\n{text}\n", file=file, flush=True)


class WriteSRT(SubtitlesWriter):
    extension: str = "srt"
    always_include_hours: bool = True
    decimal_marker: str = ","

    def write_result(self, result: dict, file: TextIO, options: dict):
        for i, (start, end, text) in enumerate(
            self.iterate_result(result, options), start=1
        ):
            print(f"{i}\n{start} --> {end}\n{text}\n", file=file, flush=True)


class WriteTSV(ResultWriter):
    """
    Write a transcript to a file in TSV (tab-separated values) format containing lines like:
    <start time in integer milliseconds>\t<end time in integer milliseconds>\t<transcript text>

    Using integer milliseconds as start and end times means there's no chance of interference from
    an environment setting a language encoding that causes the decimal in a floating point number
    to appear as a comma; also is faster and more efficient to parse & store, e.g., in C++.
    """

    extension: str = "tsv"

    def write_result(self, result: dict, file: TextIO, options: dict):
        print("start", "end", "text", sep="\t", file=file)
        for segment in result["segments"]:
            print(round(1000 * segment["start"]), file=file, end="\t")
            print(round(1000 * segment["end"]), file=file, end="\t")
            print(segment["text"].strip().replace("\t", " "), file=file, flush=True)


class WriteAudacity(ResultWriter):
    """
    Write a transcript to a text file that audacity can import as labels.
    The extension used is "aud" to distinguish it from the txt file produced by WriteTXT.
    Yet this is not an audacity project but only a label file!

    Please note : Audacity uses seconds in timestamps not ms!
    Also there is no header expected.

    If speaker is provided it is prepended to the text between double square brackets [[]].
    """

    extension: str = "aud"

    def write_result(self, result: dict, file: TextIO, options: dict):
        ARROW = "	"
        for segment in result["segments"]:
            print(segment["start"], file=file, end=ARROW)
            print(segment["end"], file=file, end=ARROW)
            print(
                (("[[" + segment["speaker"] + "]]") if "speaker" in segment else "")
                + segment["text"].strip().replace("\t", " "),
                file=file,
                flush=True,
            )


class WriteJSON(ResultWriter):
    extension: str = "json"

    def write_result(self, result: dict, file: TextIO, options: dict):
        json.dump(result, file, ensure_ascii=False)


class WriteASS(ResultWriter):
    extension: str = "ass"

    def __call__(self, result: dict, audio_path: str, options: dict):
        audio_basename = os.path.basename(audio_path)
        audio_basename = os.path.splitext(audio_basename)[0]
        output_path = os.path.join(
            self.output_dir, f"{audio_basename}.{self.extension}"
        )  # 在这里修改输出文件名
        with open(output_path, "w", encoding="utf-8") as f:
            self.write_result(result, file=f, options=options)

    def write_result(self, result: dict, file: TextIO, options: dict):
        # 写入ASS文件头部信息
        file.write("[Script Info]\n")
        file.write("Title: Generated by ResultWriter\n")
        file.write("ScriptType: v4.00+\n")
        file.write("WrapStyle: 0\n")
        file.write("\n")

        # 写入样式信息
        file.write("[V4+ Styles]\n")
        file.write(
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        )
        file.write(
            "Style: 仓耳今楷,仓耳今楷03 W04,18,&H00F7C34F,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0.9,1,2,5,5,10,1\n"
        )
        file.write("\n")

        # 写入事件
        file.write("[Events]\n")
        file.write(
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )

        for segment in result["segments"]:
            start_time = self.format_timestamp(segment["start"])
            end_time = self.format_timestamp(segment["end"])
            text = segment["text"]
            file.write(f"Dialogue: 0,{start_time},{end_time},仓耳今楷,,0,0,0,,{text}\n")

    def format_timestamp(self, seconds: float):
        # 转换时间格式为ASS文件所需的"hh:mm:ss.cc"形式
        milliseconds = round(seconds * 100)
        minutes, milliseconds = divmod(milliseconds, 6000)
        hours, minutes = divmod(minutes, 60)
        seconds, centiseconds = divmod(milliseconds, 100)

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


class WriteASS4T(ResultWriter):
    extension: str = "ass"

    def __call__(self, result: dict, audio_path: str, options: dict):
        audio_basename = os.path.basename(audio_path)
        audio_basename = os.path.splitext(audio_basename)[0]
        output_path = os.path.join(
            self.output_dir, f"{audio_basename}_{lang}.{self.extension}"
        )  # 在这里修改输出文件名
        with open(output_path, "w", encoding="utf-8") as f:
            self.write_result(result, file=f, options=options)

    def write_result(self, result: dict, file: TextIO, options: dict):
        # 写入ASS文件头部信息
        file.write("[Script Info]\n")
        file.write("Title: Generated by ResultWriter\n")
        file.write("ScriptType: v4.00+\n")
        file.write("WrapStyle: 0\n")
        file.write("\n")

        # 写入样式信息
        file.write("[V4+ Styles]\n")
        file.write(
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        )
        # file.write(
        #     f"Style: 仓耳今楷,仓耳今楷03 W04,18,&H00F7C34F,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0.9,1,2,5,5,10,1\n"
        # )
        file.write(f"Style: {ASS_STYLE}\n")
        file.write("\n")

        # 写入事件
        file.write("[Events]\n")
        file.write(
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )

        for segment in result["segments"]:
            start_time = self.format_timestamp(segment["start"])
            end_time = self.format_timestamp(segment["end"])
            text = segment["translation"]
            file.write(f"Dialogue: 0,{start_time},{end_time},仓耳今楷,,0,0,0,,{text}\n")

    def format_timestamp(self, seconds: float):
        # 转换时间格式为ASS文件所需的"hh:mm:ss.cc"形式
        milliseconds = round(seconds * 100)
        minutes, milliseconds = divmod(milliseconds, 6000)
        hours, minutes = divmod(minutes, 60)
        seconds, centiseconds = divmod(milliseconds, 100)

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def get_writer(
    output_format: str, output_dir: str
) -> Callable[[dict, TextIO, dict], None]:
    if ADD_TRANSITION:
        writers = {
            "txt": WriteTXT,
            "vtt": WriteVTT,
            "srt": WriteSRT,
            "tsv": WriteTSV,
            "json": WriteJSON,
            "ass": WriteASS,
            "srt4t": WriteSRT4T,
            "ass4t": WriteASS4T,
        }
    else:
        writers = {
            "txt": WriteTXT,
            "vtt": WriteVTT,
            "srt": WriteSRT,
            "tsv": WriteTSV,
            "json": WriteJSON,
            "ass": WriteASS,
        }
    optional_writers = {
        "aud": WriteAudacity,
    }

    if output_format == "all":
        all_writers = [writer(output_dir) for writer in writers.values()]

        def write_all(result: dict, file: TextIO, options: dict):
            for writer in all_writers:
                writer(result, file, options)

        return write_all

    if output_format in optional_writers:
        return optional_writers[output_format](output_dir)
    return writers[output_format](output_dir)
