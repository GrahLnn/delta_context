import os, toml, tomllib


def save_cache(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        toml.dump(data, f)


def load_cache(filename):
    with open(filename, "rb") as f:
        return tomllib.load(f)


def check_processing_state(cpath, fid, key, func):
    info = load_cache(f"{cpath}/{fid}.toml")
    # TODO 检测键状态然后执行func然后存储


def load_channels(filename="asset/channels.toml"):
    """加载频道数据"""
    with open(filename, "r", encoding="utf-8") as f:
        return toml.load(f)


def get_or_cache(cache_path, compute_fn):
    """从缓存中获取或计算并缓存结果"""
    if os.path.exists(cache_path):
        return load_cache(cache_path)
    result = compute_fn()
    save_cache(result, cache_path)
    return result


def save_tasklist():
    # 直接访问全局变量
    from .video_comment import comment_tasks

    if not comment_tasks:
        # 如果列表为空，删除文件（如果存在）
        try:
            if os.path.exists("cache/comment_task.toml"):
                os.remove("cache/comment_task.toml")
        except Exception as e:
            print("Error deleting tasks file:", e)
        return

    try:
        with open("cache/comment_task.toml", "w") as toml_file:
            toml.dump({"comment_task": comment_tasks}, toml_file)
    except Exception as e:
        print("Error saving tasks:", e)
