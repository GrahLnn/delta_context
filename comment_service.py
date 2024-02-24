from flask import Flask, request, jsonify
import threading
import toml
import os
from src.utils.video_comment import comment_summary_to_video

app = Flask(__name__)
comment_tasks = []
thread_lock = threading.Lock()
tasks_file = "cache/tasks.toml"


def load_tasks():
    global comment_tasks
    if os.path.exists(tasks_file):
        with open(tasks_file, "r") as f:
            data = toml.load(f)
            comment_tasks = data.get("tasks", [])
        os.remove(tasks_file)


def save_tasks():
    if not comment_tasks:
        return
    with open(tasks_file, "w") as f:
        toml.dump({"tasks": comment_tasks}, f)


@app.route("/add_task", methods=["POST"])
def add_task():
    # task = request.json
    # with thread_lock:
    #     comment_tasks.append(task)
    return jsonify({"message": "Task added successfully"}), 200


@app.after_request
def on_response(response):
    if response.status_code != 500:
        save_tasks()
    return response


# 加载任务并启动守护线程


if __name__ == "__main__":
    load_tasks()
    threading.Thread(
        target=comment_summary_to_video, args=(comment_tasks, thread_lock), daemon=True
    ).start()
    app.run(debug=True, port=5001)
