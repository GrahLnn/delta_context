from .LLM_utils import chat


async def get_summary_text(vid, timed_texts):
    vid = "zNVQfWC_evg"
    # timed_texts, lang = parse_timed_texts_and_lang(vid)

    chapters = await summarize(
        vid="vid",
        # trigger="trigger",
        timed_texts=timed_texts,
        lang="en",
        openai_api_key=KEY_OPENAI_API_KEY,
    )
    new_chapters = []
    for idx, c in enumerate(chapters):
        if idx == 0:
            new_chapters.append(c)
            continue
        if c.start - new_chapters[-1].start <= 60:
            new_chapters[-1].summary += f"\n{c.summary}"
        else:
            new_chapters.append(c)

    # Do translate && rewrite the season title
    new_new_chapters = []
    for c in new_chapters:
        tr = await translate(vid, "aabb", "zh-Hans", c, KEY_OPENAI_API_KEY)
        new_new_chapters.append(tr)
        time_format = timedelta(seconds=int(c.start))
        print(time_format, tr.chapter, tr.summary)
        print("--------------------------")
        # print(c)

    return new_new_chapters
