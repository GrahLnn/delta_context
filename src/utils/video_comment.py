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


def split_long_comment(comment: list):
    i = 0
    while i < len(comment):
        if len(comment[i]) > 900:
            # 按\n分割当前项
            parts = comment[i].split("\n")
            new_item = []  # 用来存储新生成的项，长度不超过900
            current_length = 0  # 当前new_item的长度
            for part in parts:
                part_length = len(part) + 1  # 加上一个换行符的长度
                # 检查是否加上这部分会超过900
                if current_length + part_length > 900:
                    # 如果new_item不为空，将其作为一个新项添加到res中，并重置new_item
                    if new_item:
                        comment[i] = "\n".join(new_item)
                        new_item = [part]  # 将当前部分作为新项的开始
                        current_length = part_length
                        # 将剩余的部分插入到下一个位置，如果是最后一项，则追加到列表末尾
                        if i + 1 < len(comment):
                            comment.insert(i + 1, "\n".join(parts[parts.index(part) :]))
                        else:
                            comment.append("\n".join(parts[parts.index(part) :]))
                        break  # 跳出循环，因为剩余部分已经作为新项处理
                    else:
                        # 如果new_item为空，说明当前部分本身就超过900，直接作为新项处理
                        if i + 1 < len(comment):
                            comment[i + 1] = part + "\n" + comment[i + 1]
                        else:
                            comment.append(part)
                        break  # 当前项处理完毕
                else:
                    new_item.append(part)
                    current_length += part_length
            else:
                # 如果没有超出部分，更新当前项
                comment[i] = "\n".join(new_item)
        i += 1  # 移动到下一个项进行检查
    return comment


def send_summary(summary, oid):
    summary_groups = split_summary(summary)
    summary_groups = ["\n\n".join(s) for s in summary_groups]
    summary_groups = split_long_comment(summary_groups)
    rpids = []
    for idx, s in enumerate(summary_groups):
        rpid = sync(send_comment(s, oid=oid, root=rpids[0] if idx != 0 else None))
        rpids.append(rpid)
        if idx == 0:
            time.sleep(5)
            acomment = comment.Comment(
                oid, comment.CommentResourceType.VIDEO, rpid, credential=credential
            )
            sync(acomment.pin())
        time.sleep(2)


async def video_info(bvid):
    v = video.Video(bvid=bvid, credential=credential)
    await v.get_info()
    # print(info)


def comment_summary_to_video():
    while True:
        task = None

        with thread_lock:
            if comment_tasks:
                task = comment_tasks.pop(0)

        if task is None:
            time.sleep(100)  # 等待一段时间
            continue

        try:
            current_time = datetime.now().timestamp()
            if current_time - task.get("time", current_time) < 600:
                print("Waiting to send summary", task["bvid"])

                with thread_lock:
                    comment_tasks.append(task)  # 将任务重新放回列表
                time.sleep(600)
                continue  # 跳过当前循环的剩余部分

            # 检查是否可以同步视频信息
            try:
                sync(video_info(task["bvid"]))
            except Exception as sync_error:
                print("Error syncing video info", task["bvid"], sync_error)
                # 更新任务时间并重新放回列表
                task["time"] = datetime.now().timestamp()
                if "62002" not in str(sync_error):
                    with thread_lock:
                        comment_tasks.append(task)
                continue  # 跳过当前循环的剩余部分

            # 发送摘要
            try:
                send_summary(task["summary"], task["aid"])
                print("----------------------")
                print("Send summary comment success", task["bvid"])
                print("----------------------")
            except Exception as send_error:
                print("Error sending summary", task["bvid"], send_error)
                # 更新任务时间并重新放回列表
                task["time"] = datetime.now().timestamp()
                if "12051" not in str(send_error):
                    with thread_lock:
                        comment_tasks.append(task)
                else:
                    print("ignoire this task", task["bvid"])

        except Exception as e:
            print("Error processing task", task["bvid"], "Error:", e)
            # 更新任务时间并重新放回列表
            task["time"] = datetime.now().timestamp()
            with thread_lock:
                comment_tasks.append(task)
