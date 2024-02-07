from .text_utils import clean_string, find_overlap
from .examine_auxiliary_utils import generate_segments

min_length_threshold = 10


def examine_translates(seg_translates):
    for idx, item in enumerate(seg_translates):
        if idx >= 1:
            sub_str_similarity = []
            for i in range(max(0, idx - 5), idx):
                if (
                    clean_string(item) in clean_string(seg_translates[i])
                    and len(item) > min_length_threshold
                ):
                    overlap = find_overlap(seg_translates[i], item)
                    if overlap and len(overlap) / len(item) > 0.85:
                        seg_translates[i] = seg_translates[i].replace(overlap, "", 1)
                    # display(Markdown(f"[{i}] {seg_translates[i]}\n\n[{idx}] {item}"))
            formatted_values = [f"{value*100:.2f}%" for value in sub_str_similarity]
    return seg_translates


def rebuild_result_sentences(result, seg_transcripts, seg_translates):
    match_words_seg = result["word_segments"]
    segments = []
    # print(len(seg_transcripts), len(seg_translates))
    for seg, tl in zip(seg_transcripts, seg_translates):
        # 更新 match_words_seg 并获取 data
        # print(seg, "\n", tl, "\n\n")
        match_words_seg, data = generate_segments(seg, tl, match_words_seg)
        segments.append(data)

    result["segments"] = segments
    return result
