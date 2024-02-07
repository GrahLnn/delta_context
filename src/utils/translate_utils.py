import os, sys, re, ast, spacy, json
from .LLM_utils import get_completion
from .text_utils import (
    encoding_str,
    get_codon_pair,
    add_subphrases,
    extract_names,
    fix_last_n_char,
)
from .list_utils import flatten_list
from difflib import SequenceMatcher
from src.env.env import target_language
from tqdm import tqdm


def replace_term(text, known_value, replacement):  # text uitls
    pattern = r"<term-{}>.*?<\/term>".format(known_value)
    clean_text = re.sub(pattern, replacement, text)
    # pattern = r"<term-{}>".format(known_value)
    # clean_text = re.sub(pattern, replacement, clean_text)
    # clean_text = clean_text.replace("</term>", "")
    return clean_text


def translate_text(val):
    #     sys_prompt = """你是一位精通简体中文的专业翻译，尤其擅长将专业学术论文或说话人转录内容翻译成浅显易懂的形式。我希望你能帮我将以下英文说话人转录段落翻译成中文，风格与科普杂志的中文版相似。

    # 规则：
    # - 翻译时要准确传达原文的事实和背景，同时考虑到口语化的特点和流畅性。
    # - 即使上意译也要保留原始段落格式，以及保留术语，例如 FLAC，JPEG 等。保留公司缩写，例如 Microsoft, Amazon 等。保留人名，例如John, Rebecca Saunders, Schubert 等。
    # - 同时要保留引用的论文，例如 [20] 这样的引用。
    # - 对于 Figure 和 Table，翻译的同时保留原有格式，例如：“Figure 1: ”翻译为“图 1: ”，“Table 1: ”翻译为：“表 1: ”。
    # - 全角括号换成半角括号，并在左括号前面加半角空格，右括号后面加半角空格。
    # - 输入格式为 Markdown 格式，输出格式也必须保留原始 Markdown 格式
    # - 以下是常见的术语词汇对应表：
    #   * Transformer -> Transformer
    #   * LLM/Large Language Model -> 大语言模型
    #   * Generative AI -> 生成式 AI

    # 策略：
    # 分两次翻译，并且打印结果：
    # 1. 根据英文内容直译中文，保持原有格式，不要遗漏任何信息
    # 2. 根据第一次直译的结果重新意译，遵守原意的前提下让内容更通俗易懂、符合中文表达习惯，但要保留原有格式不变，不要遗漏任何信息

    # 返回如下JSON格式：
    # {
    # "literal_translation": "直译结果",
    # "free_translation": "意译结果"
    # }"""

    #     response = get_completion(f'"{val}"', sys_prompt, json_output=True)
    prompt = f"""请将以下两个句子分割成多个独特的部分。句子应根据自然的语法或语义断点（如逗号、句号、连接词）进行分割。对于包含重复短语或中断的部分，如“I I'll be I Yeah. Thanks very much everyone. Thank you.”，应将其视为一个整体进行处理，以保持语义上的连贯性。如果两个语言版本的分割部分数量不一致，请调整较长的部分以匹配较短的部分。这意味着，如果一个语言版本的句子被分割成更多的部分，您需要将这些额外的部分合并，以确保每对 `sentence1` 和 `sentence2` 在数量和语义上相匹配，从而避免重复或空缺。在这个任务中，目标是保持原文的意义和结构，同时在适当的位置进行分割，以确保两种语言之间的对应关系清晰明确且意义一致。

## 用English to 简体中文作示例
user：There are years that ask questions and years that answer.

you：
直译:
有些年份是提问的年份，有些年份则是回答的年份。

意译:
有些岁月会提问，有些岁月会回答。

给定句子：

1. "{val}"
2. [Translate from sentence 1 in line with the Chinese expression habits]

请确保每个分割后的部分在 JSON 数组中都有唯一的对应条目，并保持句子的原始顺序。然后放在以下JSON格式里：
"segment_pairs": [
    {{
        "sentence1": "corresponding part from the 英文原句",
        "sentence2": "corresponding part from the 中文意译"
    }}
"""
    #     prompt = f"""请将以下两个句子分割成多个独特的部分。句子应根据自然的语法或语义断点（如逗号、句号、连接词）进行分割。对于包含重复短语或中断的部分，如“I I'll be I Yeah. Thanks very much everyone. Thank you.”，应将其视为一个整体进行处理，以保持语义上的连贯性。并毫无保留地按照中文习惯将其翻译为中文

    # 给定句子：

    # 1. "{val}"

    # 请确保每个分割后的部分在 JSON 数组中都有唯一的对应条目，并保持句子的原始顺序。然后放在以下JSON格式里：
    # "Translation": "[Translate sentence 1 in line with the Chinese expression habits]"
    # "segment_pairs": [
    #     {{
    #         "sentence1": "corresponding part from the 句子1",
    #         "sentence2": "sentence1的翻译对应"
    #     }}]
    # """
    response = get_completion(prompt, json_output=True)
    print(response)
    parsed_json = json.loads(response)

    # 检查翻译中是否包含所有所需的关键字
    # contains_all_keys = all(str(k) in response for k in finger_print_with_term.keys())

    return parsed_json


def ask_LLM_to_translate(texts):
    translates = []
    transcripts = []
    datas = []

    for idx, transcript in enumerate(tqdm(texts, desc="translat sentence")):
        # print(val)
        # val = encoding_str(codon_pair, val)
        # print(val)
        if not transcript.strip():
            continue
        response = translate_text(transcript)
        tcs = [d["sentence1"] for d in response["segment_pairs"]]
        tls = [d["sentence2"] for d in response["segment_pairs"]]

        if "" in tcs:
            print(tcs)
            print("has empty")
            sys.exit()

        # print(response)

        # for k, v in finger_print_with_term.items():
        #     response = replace_term(response, k, f" {v} ")
        #     # 原文也换
        #     val = replace_term(val, k, f"{v}")

        # response = response.replace("  ", " ")
        # response = re.sub(r"^.*?\n", "", response)
        # response = response.replace("\n", "")
        # print(response)

        data = [
            {
                "transcript": tc,
                "translation": tl,
            }
            for tc, tl in zip(tcs, tls)
        ]
        # print(len_check, ord_len_check)
        # print(val)
        # print(transcript_check)
        # print(response)

        datas.append(data)
        translates.append(tls)
        transcripts.append(tcs)
    # sys.exit()
    return flatten_list(transcripts), flatten_list(translates), flatten_list(datas)


def get_terms(text):
    # prompt = f'Extract academic nouns from the text that are domain-specific and might be challenging for someone outside the domain to understand. Extract all human names. Extract all naming. Exclude general programming terms that are foundational and widely recognized. connect with comma.\n\n"{text}"'
    # terms = get_completion(prompt, model="gpt-4")

    prompt = f'Extract academic naming from the text that are domain-specific and might be challenging for someone outside the domain to understand. connect with comma.\n\n"{text}"'
    terms = get_completion(prompt)
    return terms
