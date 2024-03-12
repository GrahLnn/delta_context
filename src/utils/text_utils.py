import re, difflib, random, spacy, copy, string
from difflib import SequenceMatcher
from spacy.matcher import Matcher
import regex

# from .list_utils import split_list


def replace_comma_in_brackets(text):
    def repl(match):
        return match.group(0).replace(",", "$$").replace("，", "^^")

    return re.sub(r"\([^)]+\)|（[^)]+）", repl, text)


def clean_string(s):
    return re.sub(r"[^a-zA-Z0-9\u4e00-\u9fa5 ]", "", s)


def back_to_comma(text):
    def repl(match):
        return match.group(0).replace("$$", ",").replace("^^", "，")

    return re.sub(r"\([^)]+\)|（[^)]+）", repl, text)


def remove_special_characters(s):
    return re.sub(r"[^a-zA-Z0-9 ]", "", s)


def clean_item(item):
    return item.replace("，", " ").replace("。", "")


def count_words(s):
    words = s.split()
    return len(words)


def longest_common_substring(s1, s2):
    tokens1 = re.findall(r"\w+", s1.lower())
    tokens2 = re.findall(r"\w+", s2.lower())
    m, n = len(tokens1), len(tokens2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_len = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if tokens1[i - 1] == tokens2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                max_len = max(max_len, dp[i][j])
            else:
                dp[i][j] = 0
    return max_len


def similarity_score(s1, s2, lcs_length):
    return (
        lcs_length / len(re.findall(r"\w+", s1.lower()))
        + lcs_length / len(re.findall(r"\w+", s2.lower()))
    ) / 2


def most_similar(strings):
    max_similarity = 0
    most_similar_pair = None
    indices = None

    for i in range(len(strings)):
        for j in range(i + 1, len(strings)):
            lcs = longest_common_substring(strings[i], strings[j])
            score = similarity_score(strings[i], strings[j], lcs)
            if score > max_similarity:
                max_similarity = score
                most_similar_pair = (strings[i], strings[j])

                indices = (i, j)

    return indices


def pure_length(s):
    return len(re.findall(r"[\u4e00-\u9fff]", s)) + len(s.split())


def split_sentence_with_ratio(ratio_sentence, non_split_sentence):
    # 列表，用于保存最终分割后的句子
    s_list = []

    # 按比例分割的索引
    split_indices = []

    # 当前的索引位置
    current_idx = 0

    # 计算每个子句在non_split_sentence中应占的比例
    total_length = sum(pure_length(s) for s in ratio_sentence)
    ratios = [pure_length(s) / total_length for s in ratio_sentence]

    non_split_words = non_split_sentence.split()
    if len(re.findall(r"[\u4e00-\u9fff]", non_split_sentence)) > 5:
        nlp = spacy.load("zh_core_web_sm")
        doc = nlp(non_split_sentence)
        non_split_words = [token.text for token in doc]

    # 计算非分割句子中每一部分的起始索引
    for ratio in ratios[:-1]:  # 最后一部分的索引不需要计算
        ratio_len = int(len(non_split_words) * ratio)
        ratio_len = ratio_len if ratio_len > 1 else 1
        current_idx += ratio_len
        split_indices.append(current_idx)

    # 使用计算出的索引分割非分割句子
    start_idx = 0
    if len(re.findall(r"[\u4e00-\u9fff]", non_split_sentence)) > 5:
        for idx in split_indices:
            s_list.append("".join(non_split_words[start_idx:idx]))
            start_idx = idx
        # 添加最后一部分
        s_list.append("".join(non_split_words[start_idx:]))
    else:
        for idx in split_indices:
            s_list.append(" ".join(non_split_words[start_idx:idx]))
            start_idx = idx
        # 添加最后一部分
        s_list.append(" ".join(non_split_words[start_idx:]))

    return s_list


def find_overlap(str1, str2):
    s = difflib.SequenceMatcher(None, str1, str2)
    match = s.find_longest_match(0, len(str1), 0, len(str2))
    if match.size > 0:
        return str1[match.a : match.a + match.size]
    return None


def parse_translate_string(re_translate):
    # 1. 将带斜杠的引号替换为占位符
    re_translate = re_translate.replace('\\"', "__DUOESCAPED_QUOTE__").replace(
        "\\'", "__SIGESCAPED_QUOTE__"
    )

    # 2. 将所有单引号替换为双引号
    re_translate = re_translate.replace("'", '"')

    # 3. 使用正则表达式匹配所有由双引号包裹的项
    matches = re.findall(r'"([^"]*)"', re_translate)

    # 4. 将占位符替换为带斜杠的双引号
    return [
        match.replace("__DUOESCAPED_QUOTE__", '"').replace("__SIGESCAPED_QUOTE__", "'")
        for match in matches
    ]


def find_common_subphrases(phrases):
    common_subphrases = set()

    for i in range(len(phrases)):
        for j in range(i + 1, len(phrases)):
            words_i = phrases[i].split()
            words_j = phrases[j].split()
            common_words = [word for word in words_i if word in words_j]
            if common_words:
                # 将交集的单词合并成一个子词组
                common_subphrase = " ".join(common_words)
                # 只有当子词组在两个词组中都出现时，才添加它
                if common_subphrase in phrases[i] and common_subphrase in phrases[j]:
                    common_subphrases.add(common_subphrase)

    return list(common_subphrases)


def add_subphrases(terms):
    sorted_terms = sorted(terms, key=len, reverse=True)
    phrases = [term for term in sorted_terms if " " in term]
    long_phrases = [term for term in sorted_terms if len(term.split(" ")) >= 5]
    for term in long_phrases:
        sorted_terms.remove(term)

    # 找到所有的公共子词组
    common_subphrases = find_common_subphrases(phrases)

    for subphrase in common_subphrases:
        if not subphrase in sorted_terms:
            sorted_terms.append(subphrase)

    return sorted_terms


def get_codon_pair(terms):
    mapper = UniqueMapper()

    [mapper.get_unique_code(term) for term in terms]  # 生成指纹
    term_with_finger_print = mapper.get_mapping()
    return term_with_finger_print


def encoding_str(codon_pair, text):
    terms = list(codon_pair.keys())
    single_words = [term for term in terms if " " not in term]
    phrases = [term for term in terms if " " in term]

    # 首先编码词组
    for term in phrases:
        pattern = r"(\b)" + re.escape(term) + r"(\b)"

        def replacement_func(match):
            return match.group(1) + f"{codon_pair[term]}" + match.group(2)

        text = re.sub(pattern, replacement_func, text)

    # 然后编码单个词
    for term in single_words:
        pattern = r"(\b)" + re.escape(term) + r"(\b)"

        def replacement_func(match):
            return match.group(1) + f"{codon_pair[term]}" + match.group(2)

        text = re.sub(pattern, replacement_func, text)

    # 避免词组和词的嵌套，解码成格式字符
    for term in terms:
        text = text.replace(codon_pair[term], f"<term-{codon_pair[term]}>{term}</term>")

    return text


class UniqueMapper:
    def __init__(self):
        # 生成一个0到999的随机序列
        self.available_numbers = list(range(1000))
        random.shuffle(self.available_numbers)
        self.mapping = {}

    def get_unique_code(self, input_string):
        # 如果输入已经有映射，则返回它
        if input_string in self.mapping:
            return self.mapping[input_string]

        # 否则，为新输入提供一个五位数，并保存映射
        unique_code = self.available_numbers.pop()
        self.mapping[input_string] = str(unique_code).zfill(3)
        return self.mapping[input_string]

    def get_mapping(self):
        return self.mapping


# 加载英语模型，确保先下载了相应的模型，例如: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")


def extract_names(text):
    doc = nlp(text)
    names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    return names


def escape_inner_quotes(t):
    # 将键值对中的双引号替换为临时占位符
    t = t.replace('\\"', '"')
    temp_placeholder = "$--$"
    s = re.sub(r'"\s*:\s*"', temp_placeholder, t)

    # 转义所有剩余的双引号，除了头尾的双引号
    #
    #
    #
    #
    #
    #
    try:
        s = s[0] + s[1:-1].replace('"', '\\"') + s[-1]
    except:
        print(s)

    # 将临时占位符替换回双引号
    s = s.replace(temp_placeholder, '": "')

    return s


def escape_inner_quotes_list(t):
    t = (
        t.replace("['", "+#")
        .replace('["', "+#")
        .replace("']", "#+")
        .replace('"]', "#+")
        .replace("', '", "###")
        .replace("', \"", "###")
        .replace("\", '", "###")
        .replace('", "', "###")
        .replace("'", "\\'")
    )
    t = t.replace("+#", "['").replace("#+", "']").replace("###", "', '")
    return t


def any_contains_chinese(lst):
    def contains_chinese(s):
        return bool(re.search("[\u4e00-\u9fa5]", s))

    return any(contains_chinese(item) for item in lst)


def append_missing_words_to_array(main_sentence, sentences_array):
    words = [word.strip(string.punctuation) for word in main_sentence.split()]
    current_word_index = 0
    new_sentences = []

    # 清除参考句中从下一句到结尾，或者拼接后最大交集部分句子
    for idx, sentence in enumerate(sentences_array):
        if idx > 0 and sentences_array[idx - 1].endswith(sentence):
            next_sentence = (
                sentences_array[idx + 1] if idx + 1 < len(sentences_array) else ""
            )
            match = (
                re.search(re.escape(next_sentence), main_sentence)
                if next_sentence
                else None
            )
            if match:
                check_sentence = main_sentence[: match.start()].strip()
            else:
                check_sentence = main_sentence
            pattern = r"\b" + re.escape(sentence) + r"\b"
            matches_check = re.findall(pattern, check_sentence)
            matches_sen_array = re.findall(
                pattern, " ".join(sentences_array[: idx + 1])
            )

            if len(matches_check) < len(matches_sen_array):
                sen_len = len(sentence)
                sentences_array[idx - 1] = sentences_array[idx - 1][:-sen_len].strip()

    for sentence in sentences_array:
        current_sentence_words = sentence.split()
        missing_words = []

        # 收集所有遗漏的单词，直到遇到当前句子的第一个单词为止
        while words[current_word_index].strip(
            string.punctuation
        ) != current_sentence_words[0].strip(string.punctuation):
            missing_words.append(words[current_word_index])
            current_word_index += 1

        missing_part = " ".join(missing_words)

        # 如果有遗漏的单词，将它们追加到上一个句子的末尾
        if missing_part:
            if missing_part[0] == ",":
                if new_sentences and new_sentences[-1][-1] == ".":
                    new_sentences[-1] = new_sentences[-1][:-1]
                new_sentences[-1] += missing_part
            else:
                if new_sentences and new_sentences[-1][-1] == ".":
                    new_sentences[-1] = (
                        new_sentences[-1][:-1] + " " + missing_part + "."
                    )
                else:
                    if new_sentences:
                        new_sentences[-1] += " " + missing_part
                    else:
                        new_sentences.append(missing_part)

        # 将当前句子追加到结果列表中
        new_sentences.append(sentence)
        current_word_index += len(current_sentence_words)

    # 如果主句的末尾还有遗漏的单词，将它们追加到最后一个句子的末尾
    if current_word_index < len(words):
        if new_sentences[-1][-1] == ".":
            new_sentences[-1] = new_sentences[-1][:-1]
        new_sentences[-1] += " " + " ".join(words[current_word_index:])

    return new_sentences


def check_and_fix_sentence(main_sentence, sentences_array, target_language_sentences):
    fix_sentence_array, index = remove_overlaps(
        main_sentence, copy.deepcopy(sentences_array)
    )
    for idx in index:
        remove_ratio = 1 - round(
            len(fix_sentence_array[idx]) / len(sentences_array[idx]), 3
        )
        # 若移除比例大于九成，则需要根据中文文本再次分配对应比例的单词
        if remove_ratio > 0.9:
            fix_sentence_array[idx : idx + 2] = split_sentence_with_ratio(
                target_language_sentences[idx : idx + 2],
                " ".join(fix_sentence_array[idx : idx + 2]).strip(),
            )
    return fix_sentence_array


def find_max_overlap(s1, s2):
    """找到两个句子之间的最大重叠"""
    overlap = ""
    # 逆序检查 s1 中的每个可能的子字符串是否是 s2 的前缀
    for i in range(len(s1)):
        # 截取 s1 中可能的重叠部分
        potential_overlap = s1[i : i + len(s2)]
        # 如果截取的字符串长度小于s2，就没有必要继续比较
        if len(potential_overlap) < len(s2):
            break
        # 检查截取的部分是否是 s2 的前缀
        if s2.startswith(potential_overlap):
            # 如果是，检查它是否是最长的重叠部分
            if len(potential_overlap) > len(overlap):
                overlap = potential_overlap
    return overlap


def remove_overlaps(main_sentence, sentences_array):
    index = []
    for idx in range(len(sentences_array) - 1):
        overlap = find_max_overlap(sentences_array[idx], sentences_array[idx + 1])
        if overlap:
            sentences_array[idx] = sentences_array[idx].replace(overlap, "").rstrip()
            index.append(idx)
    return sentences_array, index


def extract_right_parts(s, spliter="=", right_part=True):
    # 首先根据 '\n' 分割字符串
    lines = s.split("\n")

    # 然后筛选出包含 'spliter' 的行
    equal_lines = [line for line in lines if spliter in line]

    # 再根据 '=' 分割，然后取 '=' 右侧的项
    if right_part:
        right_parts = [line.split(spliter)[1].strip() for line in equal_lines]
    else:
        right_parts = [line.split(spliter)[0].strip() for line in equal_lines]
    right_parts = [
        item.replace("[", "").replace("]", "").strip("'").strip('"')
        for item in right_parts
    ]

    return right_parts


def fix_last_n_char(text, max_check_length=3):
    replace_characters_to_check = ("。", "，", "；")
    non_replace_characters_to_check = ("？", "！")
    left = ""
    # 检查最后的字符，找到指定字符集中字符的最早出现
    for i in range(
        -1, -min(max_check_length + 1, len(text) + 1), -1
    ):  # 从后向前遍历，最多max_check_length个字符
        if text[i] in replace_characters_to_check and i != -1:
            # 截取直到找到的特殊字符
            left = text[i + 1 :]
            text = text[:i] + "。"
            break
        elif text[i] in non_replace_characters_to_check and i != -1:
            # 截取直到找到的特殊字符并保留这个字符
            left = text[i + 1 :]
            text = text[:i] + text[i]
            break
    return text, left


def combine_words(word_list, base_max_length=25):
    combined_list = []
    current_string = ""
    current_length = 0

    # 首先计算整个列表的总长度
    total_length = sum(
        (
            1
            if re.match(r"[A-Za-z]+", word)
            else len(word) if not re.match(r"\W", word) else 0
        )
        for word in word_list
    )

    # 根据总长度调整每个分段的最大长度
    max_length = (
        base_max_length
        if total_length % base_max_length == 0
        else total_length
        // (
            (total_length // base_max_length)
            + (1 if total_length % base_max_length != 0 else 0)
        )
        + total_length % base_max_length
    )

    for word in word_list:
        word_length = (
            1
            if re.match(r"[A-Za-z]+", word)
            else len(word) if not re.match(r"\W", word) else 0
        )

        if current_length + word_length <= max_length:
            # 如果当前长度加上单词长度不超过限制，则添加到当前字符串
            current_string += " " + word + " " if re.match(r"[A-Za-z]+", word) else word

            current_length += word_length
        else:
            # 当前长度超过限制，将当前字符串添加到列表并重置

            combined_list.append(current_string.replace("  ", " ").strip())
            current_string = word
            current_length = word_length

    # 添加最后一个字符串，如果有的话
    if current_string:
        combined_list.append(current_string.replace("  ", " ").strip())

    return combined_list


def strip_chinese_punctuation(text, punctuation="。，；？！《》"):
    # 创建一个正则表达式模式，匹配字符串两端的指定标点符号
    pattern = f"^[{punctuation}]+|[{punctuation}]+$"

    # 使用正则表达式的 sub 方法去除这些标点符号
    return re.sub(pattern, "", text)


def extract_text_and_numbers(text):
    # 使用正则表达式找到匹配 \p{L} 或 \p{N} 的第一个字符
    start_match = regex.search(r"[\p{L}\p{N}]", text)
    # 使用正则表达式找到匹配 \p{L} 或 \p{N} 的最后一个字符
    end_match = regex.search(r"[\p{L}\p{N}](?!.*[\p{L}\p{N}])", text)

    if start_match and end_match:
        start_index = start_match.start()
        end_index = end_match.start() + 1  # 包含最后一个匹配字符
        return text[start_index:end_index]
    else:
        # 如果没有找到匹配的字符，则返回空字符串
        return ""


# import spacy
# from spacy.matcher import Matcher

# nlp = spacy.load("en_core_web_sm")

# # 定义数字和希腊字母的映射
# number_map = {
#     "one": 1,
#     "two": 2,
#     "three": 3,
#     "four": 4,
#     # ... 可以继续扩展
# }

# greek_map = {
#     "alpha": "α",
#     "beta": "β",
#     "gamma": "γ",
#     # ... 可以继续扩展
# }

# def replace_with_numbers_and_greek(doc):
#     # 使用Matcher来找到数字和希腊字母
#     matcher = Matcher(nlp.vocab)
#     matcher.add("NUMBER", [[{"LOWER": word}] for word in number_map.keys()])
#     matcher.add("GREEK", [[{"LOWER": word}] for word in greek_map.keys()])

#     matches = matcher(doc)
#     for match_id, start, end in matches:
#         span = doc[start:end]
#         if span.text.lower() in number_map:
#             span.merge()
#             span[0]._.set("number_value", number_map[span.text.lower()])
#         elif span.text.lower() in greek_map:
#             span.merge()
#             span[0]._.set("greek_value", greek_map[span.text.lower()])

#     # 生成新的文本
#     new_text = []
#     for token in doc:
#         if token._.has("number_value"):
#             new_text.append(str(token._.number_value))
#         elif token._.has("greek_value"):
#             new_text.append(token._.greek_value)
#         else:
#             new_text.append(token.text)
#     return ' '.join(new_text)

# # 测试
# text = "I have one apple and two bananas. Also, I know alpha, beta, and gamma."
# doc = nlp(text)
# new_text = replace_with_numbers_and_greek(doc)
# print(new_text)  # I have 1 apple and 2 bananas. Also , I know α , β , and γ .


def make_tag_info(diff_markdown):
    # 分割原始字符串为单词数组
    words = diff_markdown.split()
    # 创建一个字符串，其中包含单词和它们在原始字符串中的起始位置
    positions = [0]
    for word in words:
        positions.append(positions[-1] + len(word) + 1)  # 加1因为空格

    # 正则表达式匹配 <del>...</del> 和 <ins>...</ins>
    pattern = r"(<del>.*?</del>)|(<ins>.*?</ins>)"
    matches = list(re.finditer(pattern, diff_markdown, re.DOTALL))

    operations = []

    i = 0
    while i < len(matches):
        match = matches[i]
        start_pos, end_pos = match.span()

        # 将字符位置转换为单词数组中的索引
        start_idx = (
            next((i for i, pos in enumerate(positions) if pos > start_pos), len(words))
            - 1
        )
        end_idx = (
            next((i for i, pos in enumerate(positions) if pos >= end_pos), len(words))
            - 1
        ) + 1

        if (
            match.group().startswith("<del>")
            and i + 1 < len(matches)
            and matches[i + 1].group().startswith("<ins>")
        ):
            next_match = matches[i + 1]
            next_start_pos, next_end_pos = next_match.span()
            if end_pos == next_start_pos:
                # 替换操作
                operations.append(
                    {
                        "stage": "replace",
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "del_text": match.group(1)[5:-6],
                        "ins_text": next_match.group(2)[5:-6],
                    }
                )
                i += 2  # 跳过下一个匹配项
                continue
        if match.group().startswith("<del>"):
            # 删除操作
            operations.append(
                {
                    "stage": "delete",
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "del_text": match.group(1)[5:-6],
                }
            )
        elif match.group().startswith("<ins>"):
            # 插入操作
            operations.append(
                {
                    "stage": "insert",
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "ins_text": match.group(2)[5:-6],
                }
            )
        i += 1

    for idx, op in enumerate(operations):
        if op["stage"] == "insert":
            target = words[op["end_idx"] :]
            # while len(target) < 10:
            #     target = words[op["start_idx"] : operations[idx + 1]["start_idx"]]
            target = " ".join(target)
            # target = re.sub(r"<del>.*?<\/del>", "", target, flags=re.DOTALL)
            target = re.sub(r"<ins>.*?<\/ins>", "", target, flags=re.DOTALL)
            operations[idx]["id_str"] = (
                target.replace("<ins>", "")
                .replace("</ins>", "")
                .replace("<del>", "")
                .replace("</del>", "")
            )

    # print(operations)
    # for op in operations:
    #     print(words[op["start_idx"] : op["end_idx"]])
    ins_tags = re.finditer(r"<ins>.*?</ins>", diff_markdown)
    new_text = diff_markdown
    for tag in ins_tags:
        new_text = new_text.replace(tag.group(), "")
    # print(new_text)
    del_operations = [op for op in operations if op["stage"] != "insert"]

    ins_operations = [op for op in operations if op["stage"] == "insert"]
    # print(del_operations)

    opera_count = 0
    for i, text in enumerate(new_text.split()):
        if "<del>" in text:
            del_operations[opera_count]["start_idx"] = i
            del_operations[opera_count]["end_idx"] = i + len(
                del_operations[opera_count]["del_text"].split()
            )
            opera_count += 1

    operations = del_operations + ins_operations
    # for op in operations:
    #     print(op)
    # import sys

    # sys.exit()
    # print(operations)
    return operations


def find_sublist_indices(sublist, parent_list):
    # 找出子列表长度
    sublist_length = len(sublist)
    # 遍历父列表，直到剩余元素数量小于子列表长度
    for i in range(len(parent_list) - sublist_length + 1):
        # 如果从当前位置开始的子序列与子列表匹配
        if parent_list[i : i + sublist_length] == sublist:
            return i, i + sublist_length - 1
    # 如果找不到匹配的子列表，返回None
    return None


def make_words_equal(words, del_info):
    nwords = copy.deepcopy(words)
    insert_tasks = []
    for info in del_info:
        if info["stage"] == "delete":
            s_idx = info["start_idx"]
            e_idx = info["end_idx"]

            # print("delete:")
            # print(info["del_text"])
            # print([w["word"] for w in nwords[s_idx:e_idx]])
            nwords[s_idx:e_idx] = [{}] * (e_idx - s_idx)
        elif info["stage"] == "replace":
            # print("replace:")
            s_idx = info["start_idx"]
            e_idx = info["end_idx"]
            temp_words = nwords[s_idx:e_idx]
            ins_texts = info["ins_text"].split()
            ins_words = []
            for idx, word in enumerate(ins_texts):
                if idx == 0:
                    n = temp_words[0]
                    m = copy.deepcopy(n)
                    m["word"] = word
                    ins_words.append(m)
                else:
                    n = temp_words[-1]
                    m = copy.deepcopy(n)
                    m["word"] = word
                    ins_words.append(m)
            # print(nwords[s_idx:e_idx])
            nwords[s_idx] = ins_words
            if e_idx - s_idx > 1:
                nwords[s_idx + 1 : e_idx] = [{}] * (e_idx - s_idx - 1)
            # print(nwords[s_idx:e_idx])
        elif info["stage"] == "insert":
            id_str_list = info["id_str"].split()
            check_list = [w["word"].strip() for w in words]
            # print(id_str_list)
            # print(check_list)
            from redlines import Redlines

            diff = Redlines(
                " ".join(check_list), " ".join(id_str_list), markdown_style="none"
            )
            # print(diff.output_markdown)
            pattern = r"(<del>.*?</del>)|(<ins>.*?</ins>)"
            matches = list(re.finditer(pattern, diff.output_markdown, re.DOTALL))
            # print(matches)
            if len(matches) > 1:
                il, ir = matches[0].span()
                iil, iir = matches[1].span()
                item = diff.output_markdown[ir:iil]
                # print(item)
                id_str_list = item.split()
            id_s_i, id_e_i = find_sublist_indices(id_str_list, check_list)

            # print(id_s_i, id_e_i)
            n = words[id_s_i]
            # print("n", words[id_s_i])

            words_ins = []
            for string in info["ins_text"].split():
                m = copy.deepcopy(n)
                m["word"] = string
                words_ins.append(m)
            insert_tasks.append({"s_idx": id_s_i, "words": words_ins})
    # print("insert_tasks:")
    # print(insert_tasks)
    insert_tasks.sort(key=lambda x: x["s_idx"], reverse=True)
    # do insert
    for task in insert_tasks:
        # print(task["s_idx"], task["words"])
        nwords.insert(task["s_idx"], task["words"])
    # print(nwords)
    return nwords


def insert_space_within_av_bv(text):
    # 匹配 'av' 或 'bv'（不区分大小写），并在 a/b 和 v 之间插入空格
    # 这里使用 (?i) 来忽略大小写
    # 使用正则表达式的捕获组 () 来分别匹配 'a' 或 'b' 和 'v'
    # 然后在替换字符串中通过引用这些捕获组 (\1 和 \2)，并在它们之间插入空格
    pattern = r"(?i)(a|b)(v)"
    replaced_text = re.sub(pattern, r"\1 \2", text)
    return replaced_text
