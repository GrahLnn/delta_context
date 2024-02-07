import sys


def generate_segments(transcripts, translates, result_word_segments):
    words = transcripts.strip().split()
    previous_endtime = 0
    seg_words = []
    # Find a continuous match for words in result_word_segments
    for i in range(len(result_word_segments) - len(words) + 1):
        if all(
            result_word_segments[i + j]["word"] == words[j] for j in range(len(words))
        ):
            seg_words = result_word_segments[i : i + len(words)]

            # Split the list into parts before and after the matched segment
            before_matched = result_word_segments[:i]
            after_matched = result_word_segments[i + len(words) :]

            # Combine these parts to create a new list that doesn't contain the matched segment
            result_word_segments = before_matched + after_matched

            match_found = True
            break

    # print(transcripts)
    # print(words)
    # print(seg_words)
    start = None
    for word in seg_words:
        if "start" in word:
            start = word["start"]
            break

    if start is None:
        # 处理没有找到 "start" 的情况，例如可以设定一个默认值或抛出异常
        raise Exception(
            f"No 'start' key found in seg_words {seg_words} in transcripts '{transcripts}'"
        )  # 或者 raise Exception("No 'start' key found in seg_words")
    end = None
    for word in reversed(seg_words):
        if "end" in word:
            end = word["end"]
            break

    if end is None:
        # 处理没有找到 "end" 的情况，例如可以设定一个默认值或抛出异常
        raise Exception(
            f"No 'end' key found in seg_words {seg_words}"
        )  # 或者 raise Exception("No 'end' key found in seg_words")

    data = {
        "start": start,
        "end": end,
        "text": transcripts,
        "translation": translates.replace("。", "")
        .replace("，", " ")
        .replace("`", "")
        .replace("：", " ")
        .replace("；", " ")
        .replace(",", " ")
        .strip(),
        "words": seg_words,
    }

    return result_word_segments, data
