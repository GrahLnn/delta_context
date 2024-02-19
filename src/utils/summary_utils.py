from .LLM_utils import chat, Model, count_tokens, build_message, Role
import json
from .prompt_utils import (
    generate_multi_chapters_example_messages_for_16k,
    GENERATE_MULTI_CHAPTERS_TOKEN_LIMIT_FOR_16K,
    GENERATE_ONE_CHAPTER_SYSTEM_PROMPT,
    GENERATE_ONE_CHAPTER_TOKEN_LIMIT,
    SUMMARIZE_FIRST_CHAPTER_SYSTEM_PROMPT,
    SUMMARIZE_FIRST_CHAPTER_TOKEN_LIMIT,
    SUMMARIZE_NEXT_CHAPTER_SYSTEM_PROMPT,
    SUMMARIZE_NEXT_CHAPTER_TOKEN_LIMIT,
)
from sys import maxsize
import asyncio
from typing import Optional
from langcodes import Language
from datetime import timedelta
from dataclasses import dataclass

_TRANSLATION_SYSTEM_PROMPT = """
Given the following JSON object as shown below:

```json
{{
  "chapter": "text...",
  "summary": "text..."
}}
```

Translate the "chapter" field and "summary" field to language {lang} in BCP 47,
the translation should keep the same format as the original field.

Do not output any redundant explanation other than JSON.
"""


@dataclass
class TimedText:
    start: float = 0  # required; in seconds.
    duration: float = 0  # required; in seconds.
    # lang: str = "en"  # required; language code.
    text: str = ""  # required.


@dataclass
class Chapter:
    # cid: str = ""  # required.
    # vid: str = ""  # required.
    # slicer: str = ""  # required.
    # style: str = ""  # required.
    start: int = 0  # required; in seconds.
    lang: str = ""  # required; language code.
    chapter: str = ""  # required.
    summary: str = ""  # optional.
    refined: int = 0  # optional.


@dataclass
class Translation:
    start: str = ""
    lang: str = ""  # required; language code.
    chapter: str = ""  # required.
    summary: str = ""  # required.


async def _generate_multi_chapters(
    timed_texts: list[TimedText],
    lang: str,
    model: Model = Model.GPT_3_5_TURBO,
) -> list[Chapter]:
    chapters: list[Chapter] = []
    content: list[dict] = []

    for t in timed_texts:
        text = t.text.strip()
        if not text:
            continue
        content.append(
            {
                "start": int(t.start),
                "text": text,
            }
        )

    user_message = build_message(
        role=Role.USER,
        content=json.dumps(content, ensure_ascii=False),
    )

    if model == Model.GPT_3_5_TURBO:
        messages = generate_multi_chapters_example_messages_for_16k(lang=lang)
        messages.append(user_message)
        count = count_tokens(messages)
        if count >= GENERATE_MULTI_CHAPTERS_TOKEN_LIMIT_FOR_16K:
            # logger.info(
            #     f"generate multi chapters with 16k, reach token limit, vid={vid}, count={count}"
            # )  # nopep8.
            return chapters
    else:
        raise (500, f"generate multi chapters with wrong model, model={model}")

    try:
        content = await chat(
            messages=messages,
            model=model,
            top_p=0.1,
            timeout=90,
            json_output=True,
        )

        content = content["job"]

        # logger.info(f"generate multi chapters, vid={vid}, content=\n{content}")

        # FIXME (Matthew Lee) prompt output as JSON may not work (in the end).
        res: list[dict] = json.loads(content)
    except Exception:
        # logger.exception(f"generate multi chapters failed, vid={vid}")
        return chapters

    for r in res:
        chapter = r.get("outline", "").strip()
        information = r.get("information", "").strip()
        seconds = r.get("start", -1)

        if chapter and information and seconds >= 0:
            chapters.append(
                Chapter(
                    # cid=str(uuid4()),
                    # vid=vid,
                    # slicer=ChapterSlicer.OPENAI.value,
                    # style=ChapterStyle.TEXT.value,
                    start=seconds,
                    lang=lang,
                    chapter=chapter,
                    summary=information,
                )
            )

    # FIXME (Matthew Lee) prompt output may not sortd by seconds asc.
    return sorted(chapters, key=lambda c: c.start)


def _get_timed_texts_in_range(
    timed_texts: list[TimedText], start_time: int, end_time: int = maxsize
) -> list[TimedText]:
    res: list[TimedText] = []

    for t in timed_texts:
        if start_time <= t.start and t.start < end_time:
            res.append(t)

    return res


async def _generate_chapters_one_by_one(
    timed_texts: list[TimedText],
    lang: str,
) -> list[Chapter]:
    chapters: list[Chapter] = []
    timed_texts_start = 0
    latest_end_at = -1

    while True:
        texts = timed_texts[timed_texts_start:]
        if not texts:
            # logger.info(
            #     f"generate one chapter, drained, "
            #     f"vid={vid}, "
            #     f"len={len(timed_texts)}, "
            #     f"timed_texts_start={timed_texts_start}"
            # )
            break  # drained.

        system_prompt = GENERATE_ONE_CHAPTER_SYSTEM_PROMPT.format(
            start_time=int(texts[0].start),
            lang=lang,
        )
        system_message = build_message(Role.SYSTEM, system_prompt)

        content: list[dict] = []
        for t in texts:
            text = t.text.strip()
            if not text:
                continue

            temp = content.copy()
            temp.append(
                {
                    "index": timed_texts_start,
                    "start": int(t.start),
                    "text": text,
                }
            )

            user_message = build_message(
                role=Role.USER,
                content=json.dumps(temp, ensure_ascii=False),
            )

            if (
                count_tokens([system_message, user_message])
                < GENERATE_ONE_CHAPTER_TOKEN_LIMIT
            ):
                content = temp
                timed_texts_start += 1
            else:
                break  # for.

        user_message = build_message(
            role=Role.USER,
            content=json.dumps(content, ensure_ascii=False),
        )

        # logger.info(
        #     f"generate one chapter, "
        #     f"vid={vid}, "
        #     f"latest_end_at={latest_end_at}, "
        #     f"timed_texts_start={timed_texts_start}"
        # )

        try:
            content = await chat(
                messages=[system_message, user_message],
                model=Model.GPT_3_5_TURBO,
                top_p=0.1,
                timeout=90,
                json_output=True,
            )

            # logger.info(
            #     f"generate one chapter, vid={vid}, content=\n{content}"
            # )  # nopep8.

            # FIXME (Matthew Lee) prompt output as JSON may not work (in the end).
            res: dict = json.loads(content)
        except Exception:
            print("generate one chapter failed")
            break  # drained.

        chapter = res.get("outline", "").strip()
        seconds = res.get("start", -1)
        end_at = res.get("end_at")

        # Looks like it's the end and meanless, so ignore the chapter.
        if type(end_at) is not int:  # NoneType.
            print("generate one chapter, end_at is not int")
            break  # drained.

        if chapter and seconds >= 0:
            data = Chapter(
                # cid=str(uuid4()),
                # vid=vid,
                # slicer=ChapterSlicer.OPENAI.value,
                # style=ChapterStyle.MARKDOWN.value,
                start=seconds,
                lang=lang,
                chapter=chapter,
            )

            chapters.append(data)

        # Looks like it's the end and meanless, so ignore the chapter.
        # if type(end_at) is not int:  # NoneType.
        #     logger.info(f'generate chapters, end_at is not int, vid={vid}')
        #     break  # drained.

        if end_at <= latest_end_at:
            print("generate one chapter, avoid infinite loop")  # nopep8.
            latest_end_at += 5  # force a different context.
            timed_texts_start = latest_end_at
        elif end_at > timed_texts_start:
            print("generate one chapter, avoid drain early")  # nopep8.
            latest_end_at = timed_texts_start
            timed_texts_start = latest_end_at + 1
        else:
            latest_end_at = end_at
            timed_texts_start = end_at + 1

    return chapters


async def _summarize_chapter(
    chapter: Chapter,
    timed_texts: list[TimedText],
    lang: str,
):

    summary = ""
    summary_start = 0
    refined_count = 0

    while True:
        texts = timed_texts[summary_start:]
        if not texts:
            break  # drained.

        content = ""
        content_has_changed = False

        for t in texts:
            lines = (
                content + "\n" + f"[{t.text}]" if content else f"[{t.text}]"
            )  # nopep8.
            if refined_count <= 0:
                system_prompt = SUMMARIZE_FIRST_CHAPTER_SYSTEM_PROMPT.format(
                    chapter=chapter.chapter,
                    lang=lang,
                )
            else:
                system_prompt = SUMMARIZE_NEXT_CHAPTER_SYSTEM_PROMPT.format(
                    chapter=chapter.chapter,
                    summary=summary,
                    lang=lang,
                )

            system_message = build_message(Role.SYSTEM, system_prompt)
            user_message = build_message(Role.USER, lines)
            token_limit = (
                SUMMARIZE_FIRST_CHAPTER_TOKEN_LIMIT
                if refined_count <= 0
                else SUMMARIZE_NEXT_CHAPTER_TOKEN_LIMIT
            )

            if count_tokens([system_message, user_message]) < token_limit:
                content_has_changed = True
                content = lines.strip()
                summary_start += 1
            else:
                break  # for.

        # FIXME (Matthew Lee) it is possible that content not changed, simply avoid redundant requests.
        if not content_has_changed:
            print("summarize chapter, but content not changed")  # nopep8.
            break

        if refined_count <= 0:
            system_prompt = SUMMARIZE_FIRST_CHAPTER_SYSTEM_PROMPT.format(
                chapter=chapter.chapter,
                lang=lang,
            )
        else:
            system_prompt = SUMMARIZE_NEXT_CHAPTER_SYSTEM_PROMPT.format(
                chapter=chapter.chapter,
                summary=summary,
                lang=lang,
            )

        system_message = build_message(Role.SYSTEM, system_prompt)
        user_message = build_message(Role.USER, content)
        summary = await chat(
            messages=[system_message, user_message],
            model=Model.GPT_3_5_TURBO,
            top_p=0.1,
            timeout=90,
        )

        summary = summary.strip()
        chapter.summary = summary  # cache even not finished.
        refined_count += 1

    chapter.summary = summary.strip()
    chapter.refined = refined_count - 1 if refined_count > 0 else 0


async def summarize(
    timed_texts: list[TimedText],
    lang: str,
) -> tuple[list[Chapter], bool]:

    chapters: list[Chapter] = []

    # Use the "outline" and "information" fields if they can be generated in 4k.
    chapters = await _generate_multi_chapters(
        timed_texts=timed_texts,
        lang=lang,
        model=Model.GPT_3_5_TURBO,
    )
    if not chapters:
        chapters = await _generate_chapters_one_by_one(
            timed_texts=timed_texts,
            lang=lang,
        )
    if not chapters:
        raise (500, f"summarize failed, no chapters")
    tasks = []
    for i, c in enumerate(chapters):
        start_time = c.start
        end_time = (
            chapters[i + 1].start if i + 1 < len(chapters) else maxsize
        )  # nopep8.
        texts = _get_timed_texts_in_range(
            timed_texts=timed_texts,
            start_time=start_time,
            end_time=end_time,
        )
        tasks.append(
            _summarize_chapter(
                chapter=c,
                timed_texts=texts,
                lang=lang,
            )
        )

    res = await asyncio.gather(*tasks, return_exceptions=True)
    for r in res:
        if isinstance(r, Exception):
            raise (f"summarize, but has exception, e={r}")

    return chapters


async def translate(
    lang: str,
    chapter: Chapter,
) -> Optional[Translation]:

    # Avoid the same language.
    la = Language.get(lang)
    lb = Language.get(chapter.lang)
    if la.language == lb.language:
        return None

    system_prompt = _TRANSLATION_SYSTEM_PROMPT.format(lang=lang)
    system_message = build_message(Role.SYSTEM, system_prompt)
    user_message = build_message(
        Role.SYSTEM,
        json.dumps(
            {
                "chapter": chapter.chapter,
                "summary": chapter.summary,
            },
            ensure_ascii=False,
        ),
    )

    # Don't check token limit here, let it go.
    messages = [system_message, user_message]
    tokens = count_tokens(messages)
    # logger.info(
    #     f"translate, vid={vid}, cid={cid}, lang={lang}, tokens={tokens}"
    # )  # nopep8.

    content = await chat(
        messages=messages,
        model=Model.GPT_3_5_TURBO,
        top_p=0.1,
        timeout=90,
    )

    # logger.info(
    #     f"translate, vid={vid}, cid={cid}, lang={lang}, content=\n{content}"
    # )  # nopep8.

    # FIXME (Matthew Lee) prompt output as JSON may not work.
    res: dict = json.loads(content)
    chapter = res.get("chapter", "").strip()
    summary = res.get("summary", "").strip()

    # Both fields must exist.
    if (not chapter) or (not summary):
        raise (
            500,
            f"translate, but chapter or summary empty, lang={lang}",
        )  # nopep8.

    trans = Translation(
        lang=lang,
        chapter=chapter,
        summary=summary,
    )

    # insert_or_update_translation(trans)
    return trans


async def get_summary_text(timed_texts):

    chapters = await summarize(
        timed_texts=timed_texts,
        lang="en",
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
        tr = await translate("zh-Hans", c)

        time_format = timedelta(seconds=int(c.start))
        tr.start = str(time_format)

        new_new_chapters.append(
            {"chapter": tr.chapter, "summary": tr.summary, "start": tr.start}
        )

    return new_new_chapters


def get_timed_texts(subtitle_json):
    timed_texts = []
    for segment in subtitle_json["segments"]:
        timed_texts.append(
            TimedText(
                start=int(segment["start"]),
                duration=int(segment["end"]) - int(segment["start"]),
                text=segment["text"],
            )
        )
    return timed_texts
