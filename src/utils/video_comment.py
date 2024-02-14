from bilibili_api import comment, sync, video
from .bilibili_utils import get_credential
import time
from datetime import datetime

credential = get_credential()


def split_summary(summary):
    summary = [f"{s['start']} {s['chapter']}\n{s['summary']}" for s in summary]
    result = []
    current_length = 0
    current_group = []

    for s in summary:
        # 如果当前组的长度加上当前摘要的长度超过了1000个字符
        if current_length + len(s) > 1000:
            # 将当前组添加到结果列表中
            result.append(current_group)
            # 重置当前组和长度
            current_group = []
            current_length = 0

        # 将摘要添加到当前组
        current_group.append(s)
        current_length += len(s)

    # 添加最后一个组到结果列表中
    if current_group:
        result.append(current_group)

    return result


async def send_comment(summary, oid, root=None):
    c = await comment.send_comment(
        text=summary,
        oid=oid,
        type_=comment.CommentResourceType.VIDEO,
        root=root,
        credential=credential,
    )
    rpid = c["rpid"]

    return rpid


def send_summary(summary, oid):
    summary_groups = split_summary(summary)
    summary_groups = ["\n\n".join(s) for s in summary_groups]
    rpids = []
    for idx, s in enumerate(summary_groups):
        rpid = sync(send_comment(s, oid=oid, root=rpids[0] if idx != 0 else None))
        rpids.append(rpid)
        if idx == 0:
            time.sleep(2)
            acomment = comment.Comment(
                oid, comment.CommentResourceType.VIDEO, rpid, credential=credential
            )
            sync(acomment.pin())
        time.sleep(2)


async def video_info(bvid):
    v = video.Video(bvid=bvid, credential=credential)
    info = await v.get_info()
    # print(info)


def daemon_thread(task_list):
    while True:
        if task_list:
            # 获取任务并处理
            for idx, task in enumerate(task_list):
                check_time = task.get("time", None)
                if datetime.now().timestamp() - check_time < 600:
                    continue
                try:

                    sync(video_info(task["bvid"]))

                except Exception:
                    current_time = datetime.now().timestamp()
                    if current_time - task["time"] > 600:
                        task["time"] = current_time
                        continue
                send_summary(task["summary"], task["aid"])
                task_list.pop(idx)
            time.sleep(60)
        else:
            # 如果任务列表为空，则等待一段时间再次检查
            time.sleep(60)
