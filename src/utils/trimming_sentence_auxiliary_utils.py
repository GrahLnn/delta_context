import json, ast, re, copy, string, sys, spacy, difflib
from .list_utils import (
    flatten_list,
    remove_overlap,
    get_sequence_indices,
    is_sequence_increasing,
    split_ng_sentence,
)
from .text_utils import (
    clean_item,
    extract_right_parts,
    split_sentence_with_ratio,
    combine_words,
    fix_last_n_char,
    strip_chinese_punctuation,
)
from .LLM_utils import get_completion
from tqdm import tqdm
import math
from .status_utils import create_progress_bar


def check_pair(pair):
    data = json.loads(pair) if isinstance(pair, str) else json.loads(json.dumps(pair))
    target = list(data.values())
    target = flatten_list(target)
    state = []
    for idx, item in enumerate(target):
        if idx + 1 < len(target) and clean_item(target[idx + 1]) in clean_item(item):
            state.append("fail")
        elif idx + 1 < len(target) and clean_item(target[idx + 1]) not in clean_item(
            item
        ):
            state.append("pass")
        elif not item:
            state.append("none")
    return state


def fix_lost_item(check_array, origin_array, target_array):
    check_array_copy = [
        re.sub(r"[^\w\s]", "", item.lower()) for item in copy.deepcopy(check_array)
    ]
    origin_array_copy = [
        re.sub(r"[^\w\s]", "", item.lower()) for item in copy.deepcopy(origin_array)
    ]

    set1 = set(check_array_copy)
    set2 = set(origin_array_copy)

    unique_to_list1 = set1 - set2
    unique_to_list2 = set2 - set1

    result_list1 = list(unique_to_list1)
    result_list2 = list(unique_to_list2)

    lost_item, src = (
        (result_list1, check_array_copy)
        if result_list1
        else (result_list2, origin_array_copy)
    )
    lost_indx = [src.index(item) for item in lost_item]

    prompt = f"将以下句子分成{len(origin_array[lost_indx[0] - 1 : lost_indx[0] + 1])}部分并合并到数组\n{target_array[lost_indx[0] - 1]}"
    llm_answer = get_completion(prompt)

    target_array[lost_indx[0] - 1] = ast.literal_eval(
        re.findall(r"\[.*?\]", llm_answer)[-1]
    )
    # print(origin_array[lost_indx[0] - 1 : lost_indx[0] + 1])
    # print(target_array[lost_indx[0] - 1])

    return target_array


def split_sentence2list(array, target):
    massage = f'''请将前后句各项配平为数组, in the json format: [item from first sentence]: [segment from second sentence].
first sentence: {array}
second sentence: "{target}"'''
    # print(massage)
    llm_answer = get_completion(massage, json_output=True)
    # print(llm_answer)
    left = list(json.loads(array).keys())
    right = list(json.loads(llm_answer).values())
    # complete_translate = extract_right_parts(llm_answer)
    # if not complete_translate:
    #     llm_answer = llm_answer.replace("\n", " ")
    #     llm_answer = (
    #         llm_answer.replace("{", "")
    #         .replace("}", "")
    #         .replace("', ", "'\n")
    #         .replace('", ', '"\n')
    #     )
    #     complete_translate = extract_right_parts(llm_answer, spliter=":")
    return right, left


def fix_sentences(complete_transcript, translate):
    right_part, left_part = split_sentence2list(complete_transcript, translate)
    fix_translates = []

    if len(right_part) < len(complete_transcript):
        fix_translates = flatten_list(
            fix_lost_item(left_part, complete_transcript, right_part)
        )
    else:
        fix_translates = right_part
    return fix_translates


# 检查单词数是否和原文一致
def check_words_len(re_seg_tc, transcript):
    re_seg_tc_words = sum([len(item.split()) for item in re_seg_tc])
    tc_words = len(transcript.split())
    return re_seg_tc_words, tc_words


# 请将以下两个句子分割成多个独特的部分。句子应根据自然的语法或语义断点（如逗号、句号、连接词）进行分割。如果两个语言版本的分割部分数量不一致，请调整较长的部分以匹配较短的部分。这意味着，如果一个语言版本的句子被分割成更多的部分，您需要将这些额外的部分合并。在这个任务中，目标是保持原文的意义和结构，同时在适当的位置进行分割，以确保两种语言之间的对应关系清晰明确且意义一致。


def split_by_LLM(transcript, translate, model="gpt-3.5-turbo-0125"):
    prompt = f"""Please divide the following two sentences into multiple distinct parts. The division should be based on natural grammar or semantic breakpoints (such as commas, periods, conjunctions).


给定句子：

1. "{transcript}"
2. "{translate}"

请确保每个分割后的部分在 JSON 数组中都有唯一的对应条目，并保持句子的原始顺序。然后放在以下JSON格式里：
"segment_pairs": [
    {{{{
        "sentence1": "corresponding part from the 英文原句",
        "sentence2": "corresponding part from the 中文翻译"
    }}}}]
"""
    answer = get_completion(prompt, json_output=True, model=model)
    # print(answer)
    parsed_json = json.loads(answer)
    return parsed_json


def total_char_count(text):
    chinese_characters = len(re.findall(r"[\u4e00-\u9fff]", text))
    other_char = len(re.findall(r"[A-Za-z0-9]", text))
    total_count = chinese_characters + math.ceil(other_char / 3)
    return total_count


def split_sentence(transcripts: list[str], translates: list[str]):
    TARGET_LEN = 28
    # print(len(transcripts), len(translates))
    # for i, (transcript, translate) in enumerate(zip(transcripts, translates)):

    #     total_count = total_char_count(translate)
    #     if total_count > TARGET_LEN:
    #         if "，" in translate:
    #             seg_tl = translate.split("，")
    #             seg_tc = transcript.split(", ")
    #             if len(seg_tc) == len(seg_tl):
    #                 # seg_tc = [
    #                 #     item + "，" if idx < len(seg_tc) - 1 else item
    #                 #     for idx, item in enumerate(seg_tc)
    #                 # ]
    #                 # seg_tl = [
    #                 #     item + ", " if idx < len(seg_tl) - 1 else item
    #                 #     for idx, item in enumerate(seg_tl)
    #                 # ]
    #                 transcripts[i] = seg_tc
    #                 translates[i] = seg_tl
    # transcripts, translates = flatten_list(transcripts), flatten_list(translates)
    # print(len(transcripts), len(translates))
    count = 0
    while True:
        modified = False

        for i, (transcript, translate) in enumerate(zip(transcripts, translates)):
            if i > count:
                count = i
            total_count = total_char_count(translate)
            if total_count > TARGET_LEN:
                prompt = f'Please split the following sentence into multiple sentences. Do not add any characters. Then put them in a JSON array with a field named "part_list":\n"""{translate}"""'
                llm_answer = get_completion(prompt, json_output=True, temperature=1)
                la_combined = json.loads(llm_answer)["part_list"]
                la_combined = [item for item in la_combined if item]
                if len(la_combined) == 1:
                    prompt = f'将这句话分成两个均匀部分，不补充任何字符。然后放在一个字段为"part_list"的JSON数组里：\n"{translate}"'
                    llm_answer = get_completion(prompt, json_output=True)
                    la_combined = json.loads(llm_answer)["part_list"]
                    la_combined = [item for item in la_combined if item]

                new_la_combined = []

                idx = 0
                while idx < len(la_combined):
                    text = la_combined[idx]
                    text = str(text)

                    # 如果文本包含少于或等于两个中文字符，并且不是最后一个元素
                    if (
                        len(re.findall(r"[\u4e00-\u9fff]", text)) <= 2
                        and len(re.findall(r"[\u4e00-\u9fff]", text)) != 0
                        and idx < len(la_combined) - 1
                    ):
                        # 合并当前文本和下一个文本
                        new_la_combined.append(text + la_combined[idx + 1])
                        idx += 2  # 跳过下一个元素，因为它已经被合并
                    elif (
                        len(re.findall(r"[\u4e00-\u9fff]", text)) <= 2
                        and len(re.findall(r"[\u4e00-\u9fff]", text)) != 0
                        and idx == len(la_combined) - 1
                    ):
                        new_la_combined[-1] += text
                        idx += 1
                    else:
                        # 如果文本不满足合并条件，或者是最后一个元素，直接添加到新列表中
                        new_la_combined.append(text)
                        idx += 1

                if len(new_la_combined) == 1:
                    prompt = f'将这句话分成两个均匀部分，不补充任何字符。然后放在一个字段为"part_list"的JSON数组里：\n"{translate}"'
                    llm_answer = get_completion(prompt, json_output=True)
                    new_la_combined = json.loads(llm_answer)["part_list"]

                new_la = new_la_combined
                # current_length = 0
                # for idx, la in enumerate(new_la_combined):
                #     len_count = total_char_count(la)
                #     if current_length + len_count > TARGET_LEN:
                #         new_la.append("".join(new_la_combined[:idx]))
                #         new_la_combined = new_la_combined[idx:]
                #         current_length = 0
                #     current_length += len_count
                # if new_la_combined:
                #     new_la.append("".join(new_la_combined))

                transcripts[i] = split_sentence_with_ratio(new_la, transcript)
                translates[i] = new_la
                if "" in transcripts[i] or "" in translates[i]:
                    print("\n------------------- error -------------------")
                    print(transcripts[i])
                    print(translates[i])
                    raise Exception("empty string error")

                modified = True
                transcripts, translates = flatten_list(transcripts), flatten_list(
                    translates
                )

                break

        bar = create_progress_bar(count / len(transcripts))
        print(
            f"\rsplit: {bar} {count}/{len(transcripts)}|{(count / len(transcripts)*100):.2f}%",
            end="",
        )
        if not modified:
            break

    datas = [
        {"transcript": tc, "translation": tl}
        for tc, tl in zip(flatten_list(transcripts), flatten_list(translates))
    ]
    return transcripts, translates, datas


def split_stage(transcripts, translates, desc):
    seg_transcripts = []
    seg_translates = []

    for index, (transcript, translate) in enumerate(
        zip(tqdm(transcripts, desc=desc), translates)
    ):
        chinese_characters = len(re.findall(r"[\u4e00-\u9fff]", translate))

        if chinese_characters > 26:
            parsed_json = split_by_LLM(transcript, translate)
            # print(transcript)
            # print(translate)
            seg_tc = [
                d["sentence1"]
                for d in parsed_json["segment_pairs"]
                if "sentence1" in d and "sentence2" in d
            ]
            seg_tl = [
                d["sentence2"]
                for d in parsed_json["segment_pairs"]
                if "sentence2" in d and "sentence1" in d
            ]
            # checks = [
            #     d["check"]
            #     for d in parsed_json["segment_pairs"]
            #     if "sentence1" in d and "sentence2" in d
            # ]

            # print(seg_tc)

            # if "False" in checks and "" not in seg_tc and "" not in seg_tl:
            #     seg_tc = [transcript]
            #     seg_tl = [translate]
            while_count = 0
            # again_index = []
            nlp_zh = spacy.load("zh_core_web_sm")
            nlp_en = spacy.load("en_core_web_sm")

            while True:
                modified = False  # 标记是否进行了修改
                while_count += 1
                if while_count > 10:
                    raise Exception("dead loop error")

                for idx in range(len(seg_tc)):
                    if idx >= len(seg_tc):  # 防止索引超出范围
                        break

                    tc = seg_tc[idx]
                    tl = seg_tl[idx]

                    doc_zh = nlp_zh(tl)
                    zh_words = [token.text for token in doc_zh]
                    doc_en = nlp_en(tc)
                    en_words = [token.text for token in doc_en]

                    if (
                        strip_chinese_punctuation(tl).strip()
                        and tc.strip(string.punctuation).strip()
                        and len(en_words) / len(zh_words) < 0.4
                        and tl.strip(string.punctuation).strip()
                        != tc.strip(string.punctuation).strip()
                    ):
                        print("0")

                        seg_tc = [transcript]
                        seg_tl = [translate]
                        # checks = [d["check"] for d in parsed_json["segment_pairs"]]

                        modified = True
                        break

                    if len(seg_tc) > 1 and (
                        tc.strip(string.punctuation).strip()
                        == transcript.strip(string.punctuation).strip()
                    ):
                        print("1")
                        prompt = f'请将以下句子分割成多个独特的部分。然后放在一个字段为"part_list"的JSON数组里：\n"""{translate}"""'
                        llm_answer = get_completion(prompt, json_output=True)
                        la_combined = json.loads(llm_answer)["part_list"]

                        new_la_combined = []

                        idx = 0
                        while idx < len(la_combined):
                            text = la_combined[idx]
                            text = str(text)

                            # 如果文本包含少于或等于两个中文字符，并且不是最后一个元素
                            if (
                                len(re.findall(r"[\u4e00-\u9fff]", text)) <= 2
                                and len(re.findall(r"[\u4e00-\u9fff]", text)) != 0
                                and idx < len(la_combined) - 1
                            ):
                                # 合并当前文本和下一个文本
                                new_la_combined.append(text + la_combined[idx + 1])
                                idx += 2  # 跳过下一个元素，因为它已经被合并
                            elif (
                                len(re.findall(r"[\u4e00-\u9fff]", text)) <= 2
                                and len(re.findall(r"[\u4e00-\u9fff]", text)) != 0
                                and idx == len(la_combined) - 1
                            ):
                                new_la_combined[-1] += text
                                idx += 1
                            else:
                                # 如果文本不满足合并条件，或者是最后一个元素，直接添加到新列表中
                                new_la_combined.append(text)
                                idx += 1

                        seg_tl = new_la_combined
                        seg_tc = split_sentence_with_ratio(seg_tl, transcript)
                        modified = True
                        break

                    if len(seg_tc) == 1 and not seg_tc[0].strip():
                        print("1.5")
                        seg_tc = [transcript]
                        seg_tl = [translate]
                        modified = True
                        break
                    if len(seg_tc) > 1 and not strip_chinese_punctuation(tl).strip():
                        print("1.75")
                        seg_tc[idx - 1] += " " + tc
                        seg_tc.pop(idx)
                        seg_tl.pop(idx)
                        modified = True
                        break

                    # if len(seg_tc) > 1 and (
                    #     tl in seg_tl[idx - 1]
                    #     or (
                    #         difflib.SequenceMatcher(None, tl, seg_tl[idx - 1]).ratio()
                    #         > 0.9
                    #         and len(re.findall(r"[\u4e00-\u9fff]", "".join(seg_tl)))
                    #         - len(re.findall(r"[\u4e00-\u9fff]", tl))
                    #         > 2
                    #     )
                    # ):
                    #     print(
                    #         f"2:{difflib.SequenceMatcher(None, tl, seg_tl[idx - 1]).ratio()}"
                    #     )
                    #     # if (
                    #     #     tc.strip(string.punctuation).strip()
                    #     #     not in seg_tc[idx - 1].strip(string.punctuation).strip()
                    #     # ):
                    #     #     seg_tc[idx - 1] += " " + tc
                    #     # seg_tc.pop(idx)
                    #     # seg_tl.pop(idx)
                    #     seg_tc = [transcript]
                    #     seg_tl = [translate]

                    #     modified = True
                    #     break  # 跳出内部循环，重新开始遍历

                    if (
                        idx > 0
                        and tc.strip(string.punctuation).strip()
                        and tc.strip(string.punctuation).strip()
                        in seg_tc[idx - 1].strip(string.punctuation).strip()
                    ):
                        print("3")
                        # print(seg_tc)
                        # print(tc)
                        # print(seg_tc[idx - 1])
                        seg_tc[idx - 1] = (
                            seg_tc[idx - 1]
                            .strip(string.punctuation)
                            .strip()
                            .replace(tc.strip(string.punctuation).strip(), "")
                        )
                        # print(seg_tc[idx - 1])
                        # print(seg_tc)

                        # seg_tc.pop(idx)
                        # seg_tl.pop(idx)
                        modified = True
                        break
                    if (
                        len(re.findall(r"[\u4e00-\u9fff]", tl)) <= 2
                        and len(re.findall(r"[\u4e00-\u9fff]", tl)) != 0
                        and idx < len(seg_tc) - 1
                    ):
                        # print("4")
                        seg_tc[idx] += " " + seg_tc[idx + 1]
                        seg_tl[idx] += seg_tl[idx + 1]
                        seg_tc.pop(idx + 1)
                        seg_tl.pop(idx + 1)
                        modified = True
                        break
                    if (
                        len(seg_tc) > 1
                        and len(re.findall(r"[\u4e00-\u9fff]", tl)) <= 2
                        and len(re.findall(r"[\u4e00-\u9fff]", tl)) != 0
                        and idx == len(seg_tc) - 1
                    ):
                        print("5")
                        seg_tc[idx - 1] += " " + tc
                        seg_tl[idx - 1] += strip_chinese_punctuation(tl)
                        seg_tc.pop(idx)
                        seg_tl.pop(idx)
                        modified = True
                        break
                    # if len(re.findall(r"[\u4e00-\u9fff]", tl)) > 28 and len(seg_tc) > 1:
                    #     print("6")
                    #     seg_tc = [transcript]
                    #     seg_tl = [translate]
                    #     modified = True
                    #     break
                    if len(" ".join(seg_tc).split()) - len(transcript.split()) < -3:
                        seg_tc = [transcript]
                        seg_tl = [translate]
                        modified = True
                        break
                    if (
                        len(" ".join(seg_tc).split()) - len(transcript.split()) > 5
                        and idx > 0
                    ):
                        print("7")
                        # seg_tc = remove_overlap(seg_tc)
                        seg_tc = [transcript]
                        seg_tl = [translate]

                        modified = True
                        break
                    if not tc.strip(string.punctuation).strip():
                        print("8")
                        seg_tc = split_sentence_with_ratio(seg_tl, transcript)
                        modified = True
                        break
                    if len(tc.strip()) < 3 and idx < len(seg_tc) - 1:
                        print("9")
                        seg_tc[idx] += " " + seg_tc[idx + 1]
                        seg_tl[idx] += seg_tl[idx + 1]
                        seg_tc.pop(idx + 1)
                        seg_tl.pop(idx + 1)
                        modified = True
                        break
                    # if idx != 0 and strip_chinese_punctuation(
                    #     tl
                    # ) == strip_chinese_punctuation(seg_tl[idx - 1]):
                    #     print("10")
                    #     repeat_idx = []
                    #     for i in range(len(seg_tc)):
                    #         if strip_chinese_punctuation(
                    #             seg_tl[i]
                    #         ) == strip_chinese_punctuation(tl):
                    #             repeat_idx.append(i)
                    #     for i, n in enumerate(repeat_idx):
                    #         if i != 0 and n - repeat_idx[i - 1] != 1:
                    #             raise Exception("repeat idx error")
                    #     nlp = spacy.load("zh_core_web_sm")
                    #     doc = nlp(tl)
                    #     non_split_words = [token.text for token in doc]
                    #     mid_index = len(non_split_words) // 2
                    #     seg_tl[repeat_idx[0] : repeat_idx[-1] + 1] = [
                    #         "".join(non_split_words[:mid_index])
                    #         + " "
                    #         + "".join(non_split_words[mid_index:])
                    #     ]
                    #     seg_tc[repeat_idx[0] : repeat_idx[-1] + 1] = [
                    #         " ".join(seg_tc[repeat_idx[0] : repeat_idx[-1] + 1])
                    #     ]
                    #     modified = True
                    #     break

                    if idx < len(seg_tc) - 1:
                        text, left = fix_last_n_char(tl)
                        if left:
                            print("11")
                            seg_tl[idx] = text
                            seg_tl[idx + 1] = left + seg_tl[idx + 1]
                            # modified = True
                            # break
                    # if len(re.findall(r"[\u4e00-\u9fff]", tl)) > 25:
                    #     print("13")
                    #     prompt = f'将以下句子分成多个部分，然后放在一个字段为"sentence_list"的JSON数组里：\n{tl}'
                    #     llm_answer = get_completion(prompt, json_output=True)

                    #     la_combined = json.loads(llm_answer)["sentence_list"]
                    #     seg_tl[idx] = la_combined
                    #     seg_tc[idx] = split_sentence_with_ratio(la_combined, tc)
                    #     seg_tc = flatten_list(seg_tc)
                    #     seg_tl = flatten_list(seg_tl)
                    #     modified = True
                    #     break
                    if len(seg_tc) == 1:
                        print("12")
                        # la = split_ng_sentence(translate)
                        # la_combined = combine_words(la)
                        prompt = f'请将以下句子分割成多个独特的部分。然后放在一个字段为"part_list"的JSON数组里：\n"""{tl}"""'
                        llm_answer = get_completion(prompt, json_output=True)
                        la_combined = json.loads(llm_answer)["part_list"]

                        new_la_combined = []

                        idx = 0
                        while idx < len(la_combined):
                            text = la_combined[idx]
                            text = str(text)

                            # 如果文本包含少于或等于两个中文字符，并且不是最后一个元素
                            if (
                                len(re.findall(r"[\u4e00-\u9fff]", text)) <= 2
                                and len(re.findall(r"[\u4e00-\u9fff]", text)) != 0
                                and idx < len(la_combined) - 1
                            ):
                                # 合并当前文本和下一个文本
                                new_la_combined.append(text + la_combined[idx + 1])
                                idx += 2  # 跳过下一个元素，因为它已经被合并
                            elif (
                                len(re.findall(r"[\u4e00-\u9fff]", text)) <= 2
                                and len(re.findall(r"[\u4e00-\u9fff]", text)) != 0
                                and idx == len(la_combined) - 1
                            ):
                                new_la_combined[-1] += text
                                idx += 1
                            else:
                                # 如果文本不满足合并条件，或者是最后一个元素，直接添加到新列表中
                                new_la_combined.append(text)
                                idx += 1

                        seg_tl = new_la_combined
                        seg_tc = split_sentence_with_ratio(seg_tl, transcript)
                        # modified = True
                        # break

                if not modified:
                    break  # 如果没有进行任何修改，结束外部循环
            print(seg_tl)
            print(seg_tc)
            if len(seg_tl) == 1:
                nlp = spacy.load("zh_core_web_sm")
                doc = nlp(seg_tc[0])
                non_split_words = [token.text for token in doc]
                mid_index = len(non_split_words) // 2
                seg_tl = [
                    "".join(non_split_words[:mid_index])
                    + " "
                    + "".join(non_split_words[mid_index:])
                ]
            # print(seg_tc)
            # print(seg_tl)
            if not seg_tc:
                raise Exception("seg_tc is empty")
            # if len(transcript.split()) < len(" ".join(seg_tc).split()):
            #     seg_tc = remove_overlap(seg_tc)

            # sequence = get_sequence_indices(transcript, seg_tc)
            # try:
            #     judge = is_sequence_increasing(sequence)
            # except:
            #     print(sequence)
            #     print(transcript)
            #     print(seg_tc)
            #     print(seg_tl)
            #     sys.exit(1)
            # if not judge:
            #     chinese = re.findall(r"[\u4e00-\u9fff0-9]", translate)
            #     zhlen = re.findall(r"[\u4e00-\u9fff0-9]", "".join(seg_tl))
            #     print(
            #         f"list len:{len(seg_tl)} ordzh_len:{len(chinese)} zhlen:{len(zhlen)}\n",
            #         seg_tl,
            #     )
            #     print(translate)
            #     print(
            #         f"list len:{len(seg_tc)} ordtc_len:{len(transcript.split())} tclen:{len(' '.join(seg_tc).split())}\n",
            #     )
            #     print(transcript)
            #     print(
            #         seg_tc,
            #     )
            #     print(sequence)

            seg_transcripts.append(seg_tc)
            seg_translates.append(seg_tl)
        else:
            seg_transcripts.append(transcript)
            seg_translates.append(translate)
    datas = [
        {"transcript": tc, "translation": tl}
        for tc, tl in zip(flatten_list(seg_transcripts), flatten_list(seg_translates))
    ]

    # sys.exit(1)
    return (
        flatten_list(seg_transcripts),
        flatten_list(seg_translates),
        flatten_list(datas),
    )
