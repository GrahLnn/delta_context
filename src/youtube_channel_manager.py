import os, toml, yt_dlp


# Constants
CHANNELS_FILE = "channels.toml"
CACHE_DIR = "cache"


def initialize_channels_file():
    if not os.path.exists(CHANNELS_FILE):
        with open(CHANNELS_FILE, "w") as toml_file:
            data = {"channels with tid": {}}
            toml.dump(data, toml_file)


def initialize_cache_directory():
    if not os.path.exists(CACHE_DIR):
        os.mkdir(CACHE_DIR)


def get_video_urls(channel_name):
    channel_url = f"https://www.youtube.com/{channel_name}"
    ydl_opts = {
        "quiet": True,
        "extract_flat": True,
        "force_generic_extractor": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(channel_url, download=False)

        urls = []
        if result["entries"][0].get("url", None):
            urls = [entry["url"] for entry in result["entries"]]
            # print(f"{channel_name} has {len(urls)} videos")
            return urls
        else:
            for item in result["entries"]:
                if item["webpage_url"] == f"{channel_url}/videos":
                    urls = [subentrys["url"] for subentrys in item["entries"]]
                    # print(f"{channel_name} has {len(urls)} videos")
                    return urls


def main():
    initialize_channels_file()
    initialize_cache_directory()


if __name__ == "__main__":
    main()
