import inflect

from tqdm import tqdm

from .trimming_sentence_auxiliary_utils import split_sentence, split_stage


def split_stage_one(transcripts, translates):
    #     seg_transcripts = []
    #     seg_translates = []
    #     checks = []
    #     for index, (transcript, translate) in enumerate(
    #         zip(tqdm(transcripts, desc="split to small part"), translates)  # 存在两句话对一句话的情况
    #     ):
    #         prompt = f"""请将以下两个长句分割成更小的、相互对应的部分。在分割时，请注意保持每个分割部分的完整性，并确保它们在两种语言中相对应。对于包含重复短语或中断的部分，如“I I'll be I Yeah. Thanks very much everyone. Thank you.”，应将其视为一个整体进行处理，以保持语义上的连贯性。分割时，应在自然的停顿点（如逗号、句号等）进行，同时保持两种语言之间的意义相匹配。

    # 首句为英文，应分割为多个小段，并与第二句中文的相应部分相对应。请注意，中文翻译可能会对原文进行适当的调整，以保持流畅和自然，但意义上应保持一致。

    # 1. {transcript}
    # 2. {translate}

    # 在这个任务中，目标是保持原文的意义和结构，同时在适当的位置进行分割，以确保两种语言之间的对应关系清晰明确且意义一致。然后放在以下JSON格式里：
    # "segment_pairs": [
    # {{
    #     "sentence1": "corresponding part from the 英文原句",
    #     "sentence2": "corresponding part from the 中文翻译",
    #     "check": "True or False Whether the two sentences express the same content"
    # }}"""
    #         llm_answer = get_completion(prompt, json_output=True, top_p=0.8)
    #         # print(llm_answer)
    #         parsed_json = json.loads(llm_answer)

    #         seg_transcript = [d["sentence1"] for d in parsed_json["segment_pairs"]]
    #         seg_translate = [d["sentence2"] for d in parsed_json["segment_pairs"]]
    #         check = [d["check"] for d in parsed_json["segment_pairs"]]
    #         # print(seg_transcript)
    #         # print(seg_translate)
    #         # 验证里面是否没够分的
    #         # nlp_zh = spacy.load("zh_core_web_sm")
    #         # nlp_en = spacy.load("en_core_web_sm")
    #         # for idx, (tc, tl) in enumerate(zip(seg_transcript, seg_translate)):
    #         #     doc_zh = nlp_zh(tl)
    #         #     doc_en = nlp_en(tc)

    #         #     if (
    #         #         len(list(doc_zh.sents)) > 1
    #         #         and len(list(doc_en.sents)) > 1
    #         #         and len(re.findall(r"[\u4e00-\u9fff]", tl)) > 25
    #         #     ):
    #         #         parsed_json = split_by_LLM(tc, tl)
    #         #         deep_seg_transcript = [
    #         #             d["sentence1"] for d in parsed_json["segment_pairs"]
    #         #         ]
    #         #         deep_seg_translate = [
    #         #             d["sentence2"] for d in parsed_json["segment_pairs"]
    #         #         ]
    #         #         seg_transcript[idx] = deep_seg_transcript
    #         #         seg_translate[idx] = deep_seg_translate
    #         checks.append(check)
    #         seg_transcripts.append(seg_transcript)
    #         seg_translates.append(seg_translate)
    #     datas = [
    #         {"data": {"transcript": tc, "translation": tl, "check": ck}}
    #         for tc, tl, ck in zip(
    #             flatten_list(seg_transcripts),
    #             flatten_list(seg_translates),
    #             flatten_list(checks),
    #         )
    #     ]
    # sys.exit(1)
    return split_sentence(transcripts, translates)


def split_stage_two(transcripts, translates):
    # print(len(transcripts), len(translates))
    # for a, b in zip(transcripts, translates):
    #     print(a, "\n", b, "\n\n")
    # sys.exit(1)

    #     seg_transcripts = []
    #     seg_translates = []
    #     datas = []
    #     for idx, (transcript, translate) in enumerate(
    #         zip(tqdm(transcripts, desc="split long part"), translates)
    #     ):
    #         chinese_characters = len(re.findall(r"[\u4e00-\u9fff]", transcript))

    #         if chinese_characters > 25:
    #             tcs = re.split(", |: |; ", transcript)
    #             tls = re.split("，|：|；|。", translate)
    #             # print(f"[{idx}] {len(tcs)}|{len(tls)}\n", tcs, "\n", tls, "\n")

    #             tcs = [item for item in tcs if item]
    #             tls = [item for item in tls if item]
    #             tls = concatenate_sentences_en_end(tls)
    #             tls = [item for item in tls if item]

    #             if len(tcs) == len(tls):
    #                 for seg_tc, seg_tl in zip(tcs, tls):
    #                     char_ratio = round(
    #                         len(re.findall(r"[\u4e00-\u9fff]", seg_tl))
    #                         / len(seg_tc.split()),
    #                         3,
    #                     )
    #                     # print(char_ratio, tc, tl)

    #                     if char_ratio > 3:
    #                         tcs = [" ".join(tcs)]
    #                         # print(f"[{idx}] {len(tcs)}|{len(tls)}\n", tcs, "\n", tls, "\n")
    #                         break

    #             if len(tcs) == len(tls):
    #                 seg_transcripts.append(tcs)
    #                 seg_translates.append(tls)

    #                 # print(tcs, "\n", tls, "\n")
    #             elif len(tls) == 1:
    #                 seg_transcripts.append(" ".join(tcs))
    #                 seg_translates.append(tls)
    #             elif len(tcs) == 1:
    #                 seg_transcripts.append(tcs)
    #                 seg_translates.append("".join(tls))

    #             else:
    #                 prompt = f"""请将以下两个句子分割成多个部分，并将结果放置在一个键为 "segment_pairs" 的 JSON 数组内。每个数组元素是一个对象，包含两个字段：`sentence1` 和 `sentence2`，分别表示英文和中文句子的相应分割部分。句子应根据自然的语法或语义断点（例如逗号，顿号，连接词等）进行分割。重要的是，如果两个语言版本的分割部分数量不一致，您应该根据数量较少的一方调整另一方的分割。这意味着，如果一个语言版本的句子分割成更多的部分，您应将这些额外的部分合并，以确保每个 `sentence1` 与 `sentence2` 的对应部分在数量上一致。这样做是为了确保对应部分在语义上尽可能地匹配且避免空值。

    # 给定句子：

    # 1. "{transcript}"
    # 2. "{translate}"

    # 请确保每个分割后的部分在 JSON 数组中都有对应的条目，并保持句子的原始顺序。同时，若两个语言的分段数量不一致，应适当调整以保持语义的一致性。
    # """
    #                 # prompt1 = f"请将以下两个句子结对分割成多个部分\n\n1. {transcript}\n2. {translate}"
    #                 # llm_seg_tc = get_completion(prompt1)

    #                 # prompt2 = f'{prompt1}\n\n{llm_seg_tc}\n\n放置在键为"segment_pairs"的JSON数组内，每个数组元素是一个对象，包含两个字段：`sentence1` 和 `sentence2`，分别表示英文和中文句子的相应分割部分。若两部分长度不等，则根据少的部分找出多的部分合并\n请将以下两个句子分割成多个部分，并将结果放置在一个键为 "segment_pairs" 的 JSON 数组内。每个数组元素是一个对象，包含两个字段：`sentence1` 和 `sentence2`，分别表示英文和中文句子的相应分割部分。句子应根据自然的语法或语义断点（例如逗号，顿号，连接词等）进行分割。重要的是，如果两个语言版本的分割部分数量不一致，您应该根据数量较少的一方调整另一方的分割。这意味着，如果一个语言版本的句子分割成更多的部分，您应将这些额外的部分合并，以确保每个 `sentence1` 与 `sentence2` 的对应部分在数量上一致。这样做是为了确保对应部分在语义上尽可能地匹配。'
    #                 llm_seg_tc = get_completion(prompt, json_output=True)

    #                 print(llm_seg_tc)
    #                 parsed_json = json.loads(llm_seg_tc)
    #                 seg_tc = [d["sentence1"] for d in parsed_json["segment_pairs"]]
    #                 seg_tl = [d["sentence2"] for d in parsed_json["segment_pairs"]]

    #                 chinese = re.findall(r"[\u4e00-\u9fff]", translate)
    #                 zhlen = re.findall(r"[\u4e00-\u9fff]", "".join(seg_tl))
    #                 while True:
    #                     modified = False  # 标记是否进行了修改
    #                     for idx in range(len(seg_tc)):
    #                         if idx >= len(seg_tc):  # 防止索引超出范围
    #                             break

    #                         tc = seg_tc[idx]
    #                         tl = seg_tl[idx]

    #                         if not tl:
    #                             seg_tc[idx - 1] += " " + tc
    #                             seg_tc.pop(idx)
    #                             seg_tl.pop(idx)
    #                             modified = True
    #                             break  # 跳出内部循环，重新开始遍历
    #                         if idx > 1 and (tl in seg_tl[idx - 1] or tl in seg_tl[idx - 2]):
    #                             seg_tc[idx - 1] += " " + tc
    #                             seg_tc.pop(idx)
    #                             seg_tl.pop(idx)
    #                             modified = True
    #                             break  # 跳出内部循环，重新开始遍历
    #                         if (
    #                             len(re.findall(r"[\u4e00-\u9fff]", tl)) <= 2
    #                             and idx < len(seg_tc) - 1
    #                         ):
    #                             seg_tc[idx] += " " + seg_tc[idx + 1]
    #                             seg_tl[idx] += seg_tl[idx + 1]
    #                             seg_tc.pop(idx + 1)
    #                             seg_tl.pop(idx + 1)
    #                             modified = True
    #                             break
    #                         if (
    #                             len(re.findall(r"[\u4e00-\u9fff]", tl)) <= 2
    #                             and idx == len(seg_tc) - 1
    #                         ):
    #                             seg_tc[idx - 1] += " " + tc
    #                             seg_tl[idx - 1] += tl.strip(string.punctuation)
    #                             seg_tc.pop(idx)
    #                             seg_tl.pop(idx)
    #                             modified = True
    #                             break
    #                         if (
    #                             abs(len(" ".join(seg_tc).split()) - len(transcript.split()))
    #                             > 5
    #                         ):
    #                             seg_tc = remove_overlap(seg_tc)

    #                             modified = True
    #                             break

    #                     if not modified:
    #                         break  # 如果没有进行任何修改，结束外部循环
    #                 if len(transcript.split()) < len(" ".join(seg_tc).split()):
    #                     seg_tc = remove_overlap(seg_tc)

    #                 sequence = get_sequence_indices(transcript, seg_tc)
    #                 try:
    #                     judge = is_sequence_increasing(sequence)
    #                 except:
    #                     print(sequence)
    #                     print(transcript)
    #                     print(seg_tc)
    #                     print(seg_tl)
    #                     sys.exit(1)
    #                 if not judge:
    #                     print(
    #                         f"list len:{len(seg_tl)} ordzh_len:{len(chinese)} zhlen:{len(zhlen)}\n",
    #                         seg_tl,
    #                     )
    #                     print(translate)
    #                     print(
    #                         f"list len:{len(seg_tc)} ordtc_len:{len(transcript.split())} tclen:{len(' '.join(seg_tc).split())}\n",
    #                     )
    #                     print(transcript)
    #                     print(
    #                         seg_tc,
    #                     )
    #                     print(sequence)

    #                 seg_transcripts.append(seg_tc)
    #                 seg_translates.append(seg_tl)
    #         else:
    #             seg_transcripts.append(transcript)
    #             seg_translates.append(translate)
    #     data = [
    #         {"transcript": tc, "translation": tl}
    #         for tc, tl in zip(flatten_list(seg_transcripts), flatten_list(seg_translates))
    #     ]
    #     datas.append(data)
    # sys.exit(1)
    return split_stage(transcripts, translates, "split long part")


def get_pair(a, b):
    try:
        pair = dict(zip(a, b))
    except Exception:
        print(a)
        print(b)
    return pair


def separate_too_long_tl(seg_transcripts, seg_translates):
    # print(len(seg_transcripts), len(seg_translates))
    # nlp = spacy.load("zh_core_web_sm")
    #     new_seg_transcripts = []
    #     new_seg_translates = []
    #     for indx, (seg_t, seg_l) in enumerate(
    #         zip(
    #             tqdm(seg_transcripts, desc="separate large than 25 zh-char"), seg_translates
    #         )
    #     ):
    #         chinese_characters = len(re.findall(r"[\u4e00-\u9fff]", seg_l))

    #         # print(chinese_characters, seg_l)
    #         if chinese_characters > 25:
    #             # print(seg_t, "\n", seg_l, "\n")
    #             # print(f"separate_too_long_tl: \n{seg_t}\n[{chinese_characters}] {seg_l}\n")
    #             #             prompt = f"""任务：将两个句子分割成语义对应的段落。请根据连接词和关联词，对以下两个句子各执行一次分割。分割后，将每个句子的相应部分形成一对`segment_pairs`，并按照指定的JSON模板格式化输出。

    #             # - 英文句子："{seg_t}"
    #             # - 中文句子："{seg_l}"

    #             # 注意：
    #             # - 分割时不能重复、遗漏、提取额外内容或交换顺序。
    #             # - 确保每对`segment_pairs`在语义上相对应。

    #             # 分步骤进行：
    #             # 1. 对中文句子进行分句为不超过25个词的更小部分，返回分句
    #             # 2. 根据分好的句子赋予对应英文句子的部分
    #             # 3. 返回JSON格式"""
    #             def split_by_llm(t1, t2):
    #                 prompt = f"""请将以下两个句子分割成多个部分，并将它们以 JSON 格式组织。每个部分应在逗号、顿号等标点符号处分割，同时也要考虑连接词（如“和”，“但是”）和递进词（如“比如”，“那么”）。每个句子的对应部分应被放置在一个键为"segment_pairs"的数组内，每个数组元素是一个对象，包含两个字段：`sentence1` 和 `sentence2`，分别表示第一个和第二个句子的相应分割部分。确保每对分割后的句子片段在语义上相对应。

    # 1. "{t1}"
    # 2. "{t2}"
    # """
    #                 # print(prompt)
    #                 llm_seg_tl = get_completion(prompt, json_output=True)
    #                 parse_json = json.loads(llm_seg_tl)
    #                 return parse_json

    #             parse_json = split_by_llm(seg_t, seg_l)
    #             seg_tc = [d["sentence1"] for d in parse_json["segment_pairs"]]
    #             seg_tl = [d["sentence2"] for d in parse_json["segment_pairs"]]

    #             for idx, (sub_t, sub_l) in enumerate(zip(seg_tc, seg_tl)):
    #                 if len(re.findall(r"[\u4e00-\u9fff]", sub_l)) > 25:
    #                     parse_json = split_by_llm(sub_t, sub_l)
    #                     deep_seg_tc = [d["sentence1"] for d in parse_json["segment_pairs"]]
    #                     deep_seg_tl = [d["sentence2"] for d in parse_json["segment_pairs"]]
    #                     seg_tc[idx] = deep_seg_tc
    #                     seg_tl[idx] = deep_seg_tl
    #             seg_tc = flatten_list(seg_tc)
    #             seg_tl = flatten_list(seg_tl)
    #             idx = 0
    #             while True:
    #                 modified = False  # 标记是否进行了修改
    #                 for idx in range(len(seg_tc)):
    #                     if idx >= len(seg_tc):  # 防止索引超出范围
    #                         break

    #                     tc = seg_tc[idx]
    #                     tl = seg_tl[idx]

    #                     if not tl:
    #                         seg_tc[idx - 1] += " " + tc
    #                         seg_tc.pop(idx)
    #                         seg_tl.pop(idx)
    #                         modified = True
    #                         break  # 跳出内部循环，重新开始遍历
    #                     if idx > 1 and (tl in seg_tl[idx - 1] or tl in seg_tl[idx - 2]):
    #                         seg_tc[idx - 1] += " " + tc
    #                         seg_tc.pop(idx)
    #                         seg_tl.pop(idx)
    #                         modified = True
    #                         break  # 跳出内部循环，重新开始遍历
    #                     if (
    #                         len(re.findall(r"[\u4e00-\u9fff]", tl)) <= 2
    #                         and idx < len(seg_tc) - 1
    #                     ):
    #                         seg_tc[idx] += " " + seg_tc[idx + 1]
    #                         seg_tl[idx] += seg_tl[idx + 1]
    #                         seg_tc.pop(idx + 1)
    #                         seg_tl.pop(idx + 1)
    #                         modified = True
    #                         break
    #                     if (
    #                         len(re.findall(r"[\u4e00-\u9fff]", tl)) <= 2
    #                         and idx == len(seg_tc) - 1
    #                     ):
    #                         seg_tc[idx - 1] += " " + tc
    #                         seg_tl[idx - 1] += tl.strip(string.punctuation)
    #                         seg_tc.pop(idx)
    #                         seg_tl.pop(idx)
    #                         modified = True
    #                         break
    #                     if abs(len(" ".join(seg_tc).split()) - len(seg_t.split())) > 5:
    #                         seg_tc = remove_overlap(seg_tc)

    #                         modified = True
    #                         break

    #                 if not modified:
    #                     break  # 如果没有进行任何修改，结束外部循环

    #             print(seg_t, "\n", seg_tc)
    #             print(seg_l, "\n", seg_tl)
    #             new_seg_transcripts.extend(seg_tc)
    #             new_seg_translates.extend(seg_tl)
    #         else:
    #             new_seg_transcripts.append(seg_t)
    #             new_seg_translates.append(seg_l)

    #     # sys.exit(1)
    #     seg_transcripts = new_seg_transcripts
    #     seg_translates = new_seg_translates
    # sys.exit()
    return split_stage(
        seg_transcripts, seg_translates, "separate large than 25 zh-char"
    )


def clean_sentences(seg_transcripts, seg_translates):
    filtered_tcs = []
    filtered_tls = []

    for n, m in zip(seg_transcripts, seg_translates):
        # 只有当n和m都非空时，才添加到新的列表中
        if n != "" or m != "":
            if n == "":
                raise Exception("tcs has empty string")
            elif m == "":
                raise Exception("tls has empty string")
            else:
                filtered_tcs.append(n)
                filtered_tls.append(m)

    # 用过滤后的列表替换原列表
    seg_transcripts = filtered_tcs
    seg_translates = filtered_tls
    # 从后向前遍历列表
    for idx in range(len(seg_transcripts) - 2, -1, -1):
        current_len = len(seg_transcripts[idx].strip().split())

        # 如果当前项只有一个单词，则将它加到后一项
        if current_len == 1:
            seg_transcripts[idx + 1] = (
                seg_transcripts[idx] + " " + seg_transcripts[idx + 1]
            )
            seg_translates[idx + 1] = seg_translates[idx] + seg_translates[idx + 1]

            # 从列表中移除当前项
            seg_transcripts.pop(idx)
            seg_translates.pop(idx)
    for idx, (seg_t, seg_l) in enumerate(zip(seg_transcripts, seg_translates)):
        seg_transcripts[idx] = seg_t.strip().replace("  ", " ")

        seg_translates[idx] = seg_l.strip().replace("  ", " ")

    seg_transcripts = [normalize_text(item.strip()) for item in seg_transcripts]
    return seg_transcripts, seg_translates


def normalize_text(text):
    p = inflect.engine()
    words = text.split()
    normalized_words = [
        p.number_to_words(word) if word.isdigit() else word for word in words
    ]
    return " ".join(normalized_words)


def clean_due_space(seg_transcripts, seg_translates):
    for idx, (seg_t, seg_l) in enumerate(zip(seg_transcripts, seg_translates)):
        seg_transcripts[idx] = seg_t.strip().replace("  ", " ")
        seg_translates[idx] = seg_l.strip().replace("  ", " ")
    return seg_transcripts, seg_translates
