import os, re, toml, string, shutil


def url_type(url):
    if url.startswith(("http://", "https://", "ftp://")):
        return "remote"
    elif os.path.exists(url) or url.startswith("file:///"):
        return "local"
    else:
        return "unknown"


def sanitize_filename(title):
    # Remove invalid characters
    translator = str.maketrans("", "", string.punctuation)
    title = title.translate(translator)

    # Replace spaces with underscores
    title = title.replace(" ", "_")

    # Ensure the filename does not exceed the maximum length
    max_length = 255
    if len(title) > max_length:
        title = title[:max_length]

    return title


def save_to_toml(data, toml_file_path):
    with open(toml_file_path, "w", encoding="utf-8") as f:
        toml.dump(data, f)


def sorted_alphanumeric(data):
    """
    使用正则表达式从文件名中提取数字，并用这些数字来排序列表。
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)


def delete_contents_of_directory(dir_path):
    """
    删除指定目录下的所有内容。

    :param dir_path: 需要清空内容的目录路径。
    """
    # 确保目录存在
    if not os.path.exists(dir_path):
        print("Directory does not exist")
        return

    # 遍历目录中的每个文件/子目录
    for item_name in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item_name)

        # 如果是文件，则删除文件
        if os.path.isfile(item_path):
            os.remove(item_path)

        # 如果是目录，则删除整个目录
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
