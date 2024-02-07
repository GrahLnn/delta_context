def split_on_second_last_dot(arr):
    # 不连贯句子移动到下一项
    for i in range(len(arr) - 1):  # 遍历到倒数第二个项目
        sentences = arr[i].split(". ")
        # print(sentences)
        if len(sentences) > 1:  # 如果有超过一个句子
            # 将当前项的内容更改为除了最后一个句子的所有句子
            arr[i] = ". ".join(sentences[:-1]) + "."

            # 将最后一个句子移动到下一个项的开头
            arr[i + 1] = sentences[-1] + " " + arr[i + 1]
    # 将两项合为一项
    merged_list = []
    for i in range(0, len(arr), 2):
        merged_item = arr[i]
        if i + 1 < len(arr):
            merged_item += " " + arr[i + 1]
        merged_list.append(merged_item)

    return merged_list
