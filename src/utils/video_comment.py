from bilibili_api import comment, sync, video
from .bilibili_utils import get_credential
import time
from datetime import datetime
import sys
import threading

credential = get_credential()
comment_tasks = []
thread_lock = threading.Lock()


def split_summary(summary):
    summary = [f"{s['start']} {s['chapter']}\n{s['summary']}" for s in summary]
    result = []
    current_length = 0
    current_group = []

    for s in summary:
        # 如果当前组的长度加上当前摘要的长度超过了1000个字符
        if current_length + len(s) > 900:
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


def daemon_thread():
    while True:
        # 使用线程锁保护对全局列表的访问
        with thread_lock:
            # 检查任务列表是否为空
            if not comment_tasks:
                continue

            # 获取任务
            task = comment_tasks.pop(0)

        try:
            # 同步视频信息
            sync(video_info(task["bvid"]))

            # 检查是否满足发送条件
            current_time = datetime.now().timestamp()
            if current_time - task.get("time", current_time) < 600:
                # 如果不满足条件，稍后重试
                print("Waiting to send summary", task["bvid"])
                with thread_lock:
                    comment_tasks.append(task)  # 将任务重新放回列表
            else:
                # 满足条件，发送摘要
                send_summary(task["summary"], task["aid"])
                print("Send summary comment success", task["bvid"])

        except Exception as e:
            print("Error processing task", task["bvid"], "Error:", e)
            with thread_lock:
                comment_tasks.append(task)  # 出错时，将任务重新放回列表
