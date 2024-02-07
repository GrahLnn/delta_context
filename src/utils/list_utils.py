import string, re, copy, spacy
from .text_utils import find_max_overlap, split_sentence_with_ratio


def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list


def split_list(lst, ratio):
    split_idx = int(len(lst) * ratio)
    return lst[:split_idx], lst[split_idx:]


def get_sequence_indices(reference, segments):
    reference_words = reference.split()
    # print(reference_words)
    # reference_words_stripped = [
    #     word.strip(string.punctuation) for word in reference_words
    # ]
    reference_words_stripped = [
        word.translate(str.maketrans("", "", string.punctuation))
        for word in reference_words
    ]
    # print(reference_words_stripped)
    segment_indices = []
    segments_for_check = [
        segment.translate(str.maketrans("", "", string.punctuation))
        for segment in copy.deepcopy(segments)
    ]

    for idx, segment in enumerate(segments):
        segment_first_word = segment.split()[0].translate(
            str.maketrans("", "", string.punctuation)
        )
        # print(f"当前段落: {segment}, 首单词: {segment_first_word}")
        # if len(segment.split()) <= 1:
        #     continue

        # 清除之前段落中的重复部分
        for i in range(idx):
            segment_clean = segment.lower().translate(
                str.maketrans("", "", string.punctuation)
            )
            previous_segment_clean = (
                segments[i].lower().translate(str.maketrans("", "", string.punctuation))
            )

            # 使用正则表达式确保匹配整个单词或短语
            pattern_prev = re.compile(r"\b" + re.escape(segment_clean) + r"\b")
            match_prev = pattern_prev.search(previous_segment_clean)
            if match_prev:
                index_in_prev = match_prev.start()
                segments_for_check[i] = segments_for_check[i][:index_in_prev]

            pattern_current = re.compile(
                r"\b" + re.escape(previous_segment_clean) + r"\b"
            )
            match_current = pattern_current.search(segment_clean)
            if match_current:
                index_in_current = match_current.start()
                end_index = index_in_current + len(previous_segment_clean)
                segments_for_check[i] = segments_for_check[i][end_index:].strip()

            # 检查前句是否在当前句中
            if re.search(
                r"\b" + re.escape(previous_segment_clean) + r"\b", segment_clean
            ):
                index_in_current = segment_clean.find(previous_segment_clean)

                end_index = index_in_current + len(previous_segment_clean)
                segments_for_check[i] = segments_for_check[i][end_index:].strip()
                # print(
                #     f"前句在当前段落中: {segments[i]} => {segments_for_check[i][end_index:].strip()}"
                # )

        # 计算单词在之前段落中的出现次数
        count_word = 0
        for i in range(idx):
            pattern = r"\b" + re.escape(segment_first_word) + r"\b"
            count_word += len(re.findall(pattern, segments_for_check[i], re.IGNORECASE))

        # 查找当前段落的第一个单词在参考文本中的索引
        current_count = 0
        segment_index = -1
        for i, word in enumerate(reference_words_stripped):
            if word.lower() == segment_first_word.lower():
                if current_count == count_word:
                    segment_index = i
                    break
                current_count += 1
        # print(f"单词 '{segment_first_word}' 在之前段落中的出现次数: {count_word}")
        # print(segments_for_check)
        # print()

        if segment_index == -1:
            raise ValueError(
                f"The word '{segment_first_word}' was not found the expected number of times.\n{reference}\n{segments}"
            )
        segment_indices.append(segment_index)

    return segment_indices


def is_sequence_increasing(sequence_indices):
    # 使用 enumerate 获取索引和值
    for i, current_index in enumerate(
        sequence_indices[:-1]
    ):  # 最后一个索引没有后续的索引可以比较，所以我们不包括它
        # 检查是否有 None 值或当前索引大于或等于下一个索引
        if (
            current_index is None
            or sequence_indices[i + 1] is None
            or current_index > sequence_indices[i + 1]
        ):
            return False
    return True


def fix_sequences(reference, segments, sequence):
    reference_words = reference.split()
    status = [
        "right"
        if i is not None
        and (i == 0 or all(i > previous for previous in sequence[:idx]))
        else "wrong"
        for idx, i in enumerate(sequence)
    ]

    for idx in range(1, len(status)):
        if status[idx] == "wrong":
            status[idx - 1] = "wrong"
    # print(status)

    # 组合连续的 'wrong' 标记项
    wrong_groups = []
    current_group = []
    for idx, s in enumerate(status):
        if s == "wrong":
            current_group.append(idx)
        else:
            if current_group:
                wrong_groups.append(current_group)
                current_group = []

    if current_group:
        wrong_groups.append(current_group)
    # print(wrong_groups)
    # print(current_group)

    # 修复每个 'wrong' 组
    for group in wrong_groups:
        # 找到修复的句子的起始和结束索引
        start_index = min(sequence[i] for i in group if sequence[i] is not None)
        # print(start_index)
        end_index = (
            sequence[group[-1] + 1]
            if group[-1] + 1 < len(sequence)
            else len(reference_words)
        )
        # print(end_index)

        # 提取并修复句子
        repaired_sentence = " ".join(reference_words[start_index:end_index])

        # 为 'wrong' 标记的分割句数组项赋值修复后的句子
        for idx in group:
            segments[idx] = repaired_sentence

    return segments


def get_sequence_overlap(reference, segments):
    reference_words = reference.split()
    # 创建一个去除所有单词首尾标点的参考单词列表
    reference_words_stripped = [
        word.strip(string.punctuation) for word in reference_words
    ]
    segment_indices = []
    # print(reference_words_stripped)

    for idx, segment in enumerate(segments):
        segment_first_word = segment.split()[0].strip(string.punctuation)
        # print(segment_first_word)

        count_word = 0
        for i in range(idx):  # 若当前句子在前句的公共部分中有，则移除出现的第一个单词到这句的末尾
            segments_for_check = copy.deepcopy(segments)
            max_overlap = find_max_overlap(
                segments_for_check[i].lower().strip(string.punctuation),
                segment.lower().strip(string.punctuation),
            )
            # print(max_overlap)
            if max_overlap in segments[i].lower().strip(string.punctuation):
                index = (
                    segments_for_check[i]
                    .lower()
                    .strip(string.punctuation)
                    .find(segment.lower().strip(string.punctuation))
                )
                # print(index)
                if index != -1:
                    segments_for_check[i] = segments_for_check[i][:index]
                else:
                    segments_for_check[i] = ""
            # print(segments_for_check[i])

            count_word += len(
                re.findall(
                    r"\b" + re.escape(segment_first_word) + r"\b",
                    segments_for_check[i],
                    re.IGNORECASE,
                )
            )
        # print(count_word)
        # try:
        # 从参考句子中第一次出现的位置开始查找
        current_count = 0
        segment_index = -1
        for i, word in enumerate(reference_words_stripped):
            if word.lower() == segment_first_word.lower():
                if current_count == count_word:
                    segment_index = i
                    break
                current_count += 1

        # 检查是否找到了相应的单词
        if segment_index == -1:
            raise ValueError(
                f"The word '{segment_first_word}' was not found the expected number of times.\n{reference}\n{segments}"
            )
        segment_indices.append([segment_index, segment_index + len(segment.split())])
        # except ValueError:
        #     # 如果找不到单词，设置索引为None
        #     segment_indices.append(None)

    return segment_indices


def make_over_notion(sequence):
    over_notion = copy.deepcopy(sequence)  # 复制sequence用于标记

    for idx, item in enumerate(sequence):
        # 如果不是第一个元素，并且当前项的起始小于前一项的结束，则标记为重叠
        if idx > 0 and item[0] < sequence[idx - 1][1]:
            over_notion[idx] = "overlap"
            # 逆序遍历当前索引之前的元素
            for i in range(idx - 1, -1, -1):
                # 如果当前项的起始小于之前项的结束，标记之前的项为重叠
                if item[0] < sequence[i][1]:
                    over_notion[i] = "overlap"
                else:
                    # 如果没有重叠，则退出循环，因为前面的不会重叠
                    break
    return over_notion


def fix_overlap(
    overlap_sequence, reference_sentence, seg_sentences, translate_seg_sentences
):
    reference_words = reference_sentence.split()

    # 找到所有连续 'overlap' 的索引组
    overlap_groups = []
    current_group = []
    for idx, status in enumerate(overlap_sequence):
        if status == "overlap":
            current_group.append(idx)

        else:
            if current_group:
                overlap_groups.append(current_group)
                current_group = []
    if current_group:
        overlap_groups.append(current_group)

    # 处理每个 'overlap' 索引组
    for group in overlap_groups:
        # 获取组的起始和结束索引，以及参考句子的对应子句
        target_array = translate_seg_sentences[group[0] : group[-1] + 1]

        start_idx = overlap_sequence[group[0] - 1][1] if group[0] - 1 >= 0 else 0
        end_idx = (
            overlap_sequence[group[-1] + 1][0] - 1
            if group[-1] + 1 < len(overlap_sequence)
            else len(reference_words)
        )

        target_sentence = " ".join(reference_words[start_idx:end_idx])

        seg_sentences[group[0] : group[-1] + 1] = split_sentence_with_ratio(
            target_array, target_sentence
        )

    return seg_sentences


def split_string_around_substring(main_string, substring):
    split_index = main_string.find(substring)

    if split_index != -1:
        len_substring = len(substring)
        before = main_string[:split_index]
        after = main_string[split_index + len_substring :]
        return [before, after]
    else:
        return [main_string]


def remove_overlap(sentences):
    """
    移除数组中每个元素与其前一个元素的重叠部分。

    :param sentences: 包含句子的数组。
    :return: 处理后的句子数组。
    """
    if len(sentences) < 2:
        return sentences

    overlap_occurred = True  # 初始化重叠发生标志

    while overlap_occurred:
        overlap_occurred = False
        processed_sentences = [sentences[0]]

        for i in range(1, len(sentences)):
            prev_sentence = sentences[i - 1]
            current_sentence = sentences[i]

            # 查找重叠部分
            overlap = ""
            for j in range(min(len(prev_sentence), len(current_sentence)), 0, -1):
                if prev_sentence[-j:] == current_sentence[:j]:
                    overlap = current_sentence[:j]
                    break

            # 如果存在重叠，则设置标志为 True
            if overlap.strip():
                overlap_occurred = True

            # 移除重叠部分并添加到结果列表
            new_sentence = (
                current_sentence[len(overlap) :].strip(string.punctuation).strip()
            )
            processed_sentences.append(new_sentence)

        if processed_sentences == sentences:  # 如果处理后的句子和处理前相同，退出循环
            break

        sentences = processed_sentences

    return sentences


def split_ng_sentence(text):
    list_a = text.split()
    nlp = spacy.load("zh_core_web_sm")
    for idx, n in enumerate(list_a):
        if re.match(r"[\u4e00-\u9fa5]", n):
            print(n)
            doc = nlp(n)
            list_a[idx] = [token.text for token in doc]
    list_a = flatten_list(list_a)
    return list_a
