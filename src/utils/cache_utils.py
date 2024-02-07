import tomllib, os, toml


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
    with open(filename, "rb") as f:
        return tomllib.load(f)


def get_or_cache(cache_path, compute_fn):
    """从缓存中获取或计算并缓存结果"""
    if os.path.exists(cache_path):
        return load_cache(cache_path)
    result = compute_fn()
    save_cache(result, cache_path)
    return result
