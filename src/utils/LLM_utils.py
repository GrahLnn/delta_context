import time, os, sys, json, tiktoken, threading, functools, traceback, requests
from .net_utils import checker

# from openai import OpenAI
from asset.env.env import OPENAI_API_KEY

# from dotenv import load_dotenv
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from concurrent.futures import ThreadPoolExecutor


import httpx
from dataclasses import asdict, dataclass
from enum import unique
from strenum import StrEnum


class TokenLimit:
    GPT_3_5_TURBO = 16384
    # GPT_3_5_TURBO_16K = 16384
    GPT_4 = 8192
    GPT_4_32K = 32768


def timeout(seconds):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [
                Exception(
                    f"function [{func.__name__}] timeout [{seconds} seconds] exceeded!"
                )
            ]

            def newFunc(stop_event):
                try:
                    res[0] = func(*args, **kwargs, stop_event=stop_event)
                except Exception as e:
                    res[0] = e

            stop_event = threading.Event()
            t = threading.Thread(target=newFunc, args=(stop_event,))
            t.start()
            t.join(min(seconds, 120))

            if t.is_alive():
                stop_event.set()  # 通知线程停止
                print("set stop event")
                t.join()  # 等待线程结束
                print("join done")
                raise res[0]

            return res[0]

        return wrapper

    return decorator


# from asset.env.env import OPENAI_API_KEY

# openai = OpenAI(api_key=OPENAI_API_KEY)


class CostCalculator:
    _instance = None

    class _CostCalculator:
        def __init__(self):
            self.model_cost_input = {
                "gpt-3.5-turbo-1106": 0.001,
                "gpt-3.5-turbo-0125": 0.0005,
                "gpt-3.5-turbo": 0.0015,
                "gpt-3.5-turbo-16k": 0.003,
                "gpt-4": 0.03,
                "gpt-4-1106-preview": 0.01,
            }
            self.model_cost_output = {
                "gpt-3.5-turbo-1106": 0.002,
                "gpt-3.5-turbo-0125": 0.0015,
                "gpt-3.5-turbo": 0.002,
                "gpt-3.5-turbo-16k": 0.004,
                "gpt-4": 0.06,
                "gpt-4-1106-preview": 0.03,
            }
            self.total_cost = 0.0  # 累积费用

        def calculate(self, model, input_tokens, output_tokens):
            if (
                model not in self.model_cost_input
                or model not in self.model_cost_output
            ):
                raise ValueError(f"Unknown model: {model}")

            input_cost = self.model_cost_input[model] * input_tokens / 1000
            output_cost = self.model_cost_output[model] * output_tokens / 1000
            cost = round(input_cost + output_cost, 6)

            # 更新累积费用
            self.total_cost += cost
            return cost

        def get_total_cost(self):
            return self.total_cost

        def reset_total_cost(self):
            self.total_cost = 0.0

    def __new__(cls):
        if not CostCalculator._instance:
            CostCalculator._instance = CostCalculator._CostCalculator()
        return CostCalculator._instance


cost_calculator = CostCalculator()


def get_completion(
    prompt,
    sys_prompt="",
    model="gpt-3.5-turbo-0125",
    temperature=0,
    top_p=1,
    json_output=False,
):
    # print(sys_prompt, "\n", prompt)
    # response = openai_completion(
    #     prompt, sys_prompt, model, temperature, top_p, json_output
    # )

    mas_retry = 2
    while True:
        try:
            response = predict_no_ui_long_connection(
                prompt,
                sys_prompt,
                json_output=json_output,
                model=model,
                temperature=temperature,
            )
            break
        except Exception as e:
            if mas_retry == 0:
                raise e
            mas_retry -= 1
            sleep_time = 1
            while sleep_time < 60:
                print(
                    f"OpenAI API error, retrying...{60-sleep_time}s",
                    end="\r",
                    flush=True,
                )
                time.sleep(1)
                sleep_time += 1

    return response


# def yi_completion(
#     prompt,
#     sys_prompt="",
#     model="gpt-3.5-turbo-1106",
#     temperature=0,
#     top_p=1,
#     json_output=False,
# ):
#     model_path = "01-ai/Yi-34b-Chat"

#     tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
#     # Since transformers 4.35.0, the GPT-Q/AWQ model can be loaded using AutoModelForCausalLM.
#     model = AutoModelForCausalLM.from_pretrained(
#         model_path, device_map="auto", torch_dtype="auto"
#     ).eval()

#     if not sys_prompt:
#         messages = [
#             {"role": "user", "content": prompt},
#         ]
#     else:
#         messages = [
#             {"role": "system", "content": sys_prompt},
#             {"role": "user", "content": prompt},
#         ]

#     # Prompt content: "hi"
#     messages = [{"role": "user", "content": "hi"}]

#     input_ids = tokenizer.apply_chat_template(
#         conversation=messages,
#         tokenize=True,
#         add_generation_prompt=True,
#         return_tensors="pt",
#     )
#     output_ids = model.generate(input_ids.to("cuda"))
#     response = tokenizer.decode(
#         output_ids[0][input_ids.shape[1] :], skip_special_tokens=True
#     )


def generate_payload(
    inputs,
    history,
    system_prompt,
    stream,
    model="gpt-3.5-turbo-1025",
    json_output=False,
    temperature=0,
):
    """
    整合所有信息，生成OpenAI HTTP请求的headers和payload，为发送请求做准备。
    """
    # 验证API密钥
    # if not is_any_api_key(llm_kwargs["api_key"]):
    #     raise AssertionError(
    #         "你提供了错误的API_KEY。\n\n1. 临时解决方案：直接在输入区键入api_key，然后回车提交。\n\n2. 长效解决方案：在config.py中配置。"
    #     )

    # 选择API密钥
    api_key = OPENAI_API_KEY

    # 构建请求头部
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    # if API_ORG.startswith("org-"):
    #     headers.update({"OpenAI-Organization": API_ORG})

    # 构建消息体
    messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    # for index in range(0, len(history), 2):
    #     messages.append({"role": "user", "content": history[index]})
    #     if index + 1 < len(history):
    #         messages.append({"role": "assistant", "content": history[index + 1]})

    # 加入当前输入
    messages.append({"role": "user", "content": inputs})

    # 构建有效载荷
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": 1,
        "n": 1,
        "stream": stream,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "response_format": {"type": "json_object"} if json_output else None,
    }

    return headers, payload


MAX_RETRY = 5


def get_full_error(chunk, stream_response):
    """
    获取完整的从Openai返回的报错
    """
    while True:
        try:
            chunk += next(stream_response)
        except Exception:
            break
    return chunk


def decode_chunk(chunk):
    # 提前读取一些信息 （用于判断异常）
    chunk_decoded = chunk.decode()
    chunkjson = None
    has_choices = False
    choice_valid = False
    has_content = False
    has_role = False
    try:
        chunkjson = json.loads(chunk_decoded[6:])
        has_choices = "choices" in chunkjson
        if has_choices:
            choice_valid = len(chunkjson["choices"]) > 0
        if has_choices and choice_valid:
            has_content = "content" in chunkjson["choices"][0]["delta"]
        if has_content:
            has_content = chunkjson["choices"][0]["delta"]["content"] is not None
        if has_choices and choice_valid:
            has_role = "role" in chunkjson["choices"][0]["delta"]
    except Exception:
        pass
    return chunk_decoded, chunkjson, has_choices, choice_valid, has_content, has_role


def chat_complation(headers, payload, MAX_RETRY=5):
    retry = 0
    while True:
        try:
            # make a POST request to the API endpoint, stream=False
            sleep_time = 1
            while not checker.can_execute_request():
                # print(
                #     f"Network unstable, waiting...{sleep_time}s",
                #     end="\r",
                #     flush=True,
                # )
                time.sleep(1)
                sleep_time += 1
            endpoint = "https://api.openai.com/v1/chat/completions"
            response = requests.post(
                endpoint,
                headers=headers,
                # proxies=proxies,
                json=payload,
                stream=True,
                timeout=30,
            )
            break
        except requests.exceptions.ReadTimeout:
            retry += 1
            # traceback.print_exc()
            if retry > MAX_RETRY:
                raise TimeoutError
            if MAX_RETRY != 0:
                # print(f"\n请求超时，正在重试 ({retry}/{MAX_RETRY}) ……")
                time.sleep(5)
        except Exception as e:
            # traceback.print_exc()
            retry += 1
            if retry > MAX_RETRY:
                raise e
            if MAX_RETRY != 0:
                # print(f"\n请求异常，正在重试 ({retry}/{MAX_RETRY}) ……")
                time.sleep(5)

    stream_response = response.iter_lines()
    result = ""
    json_data = None
    while True:
        try:
            chunk = next(stream_response)
        except StopIteration:
            break
        except requests.exceptions.ConnectionError:
            chunk = next(stream_response)  # 失败了，重试一次？再失败就没办法了。
        (
            chunk_decoded,
            chunkjson,
            has_choices,
            choice_valid,
            has_content,
            has_role,
        ) = decode_chunk(chunk)
        if len(chunk_decoded) == 0:
            continue
        if not chunk_decoded.startswith("data:"):
            error_msg = get_full_error(chunk, stream_response).decode()
            if "reduce the length" in error_msg:
                raise ConnectionAbortedError("OpenAI拒绝了请求:" + error_msg)
            else:
                raise RuntimeError("OpenAI拒绝了请求：" + error_msg)
        if "data: [DONE]" in chunk_decoded:
            break  # api2d 正常完成
        # 提前读取一些信息 （用于判断异常）
        if has_choices and not choice_valid:
            # 一些垃圾第三方接口的出现这样的错误
            continue
        json_data = chunkjson["choices"][0]
        delta = json_data["delta"]
        if len(delta) == 0:
            break
        if "role" in delta:
            continue
        if "content" in delta:
            # print(delta["content"], end="", flush=True)
            result += delta["content"]

        else:
            raise RuntimeError("意外Json结构：" + delta)
    if json_data and json_data["finish_reason"] == "content_filter":
        raise RuntimeError("由于提问含不合规内容被Azure过滤。")
    if json_data and json_data["finish_reason"] == "length":
        raise ConnectionAbortedError(
            "正常结束，但显示Token不足，导致输出不完整，请削减单次输入的文本量。"
        )

    return result


def predict_no_ui_long_connection(
    inputs,
    sys_prompt="",
    history=[],
    json_output=False,
    model="gpt-3.5-turbo-0125",
    temperature=0,
):
    """
    发送至chatGPT，等待回复，一次性完成，不显示中间过程。但内部用stream的方法避免中途网线被掐。
    inputs：
        是本次问询的输入
    sys_prompt:
        系统静默prompt
    llm_kwargs：
        chatGPT的内部调优参数
    history：
        是之前的对话列表
    observe_window = None：
        用于负责跨越线程传递已经输出的部分，大部分时候仅仅为了fancy的视觉效果，留空即可。observe_window[0]：观测窗。observe_window[1]：看门狗
    """
    if model == "gpt-4":
        model = "gpt-3.5-turbo-0125"

    headers, payload = generate_payload(
        inputs,
        history,
        system_prompt=sys_prompt,
        stream=True,
        model=model,
        json_output=json_output,
        temperature=temperature,
    )
    result = chat_complation(headers, payload)
    # print()
    enc = tiktoken.encoding_for_model(model)
    count_input_token = len(enc.encode(inputs + " " + sys_prompt))
    count_output_token = len(enc.encode(result))
    # print("token:" + str(count_input_token + count_output_token))
    cost_calculator.calculate(model, count_input_token, count_output_token)

    return result


# def openai_completion(
#     prompt,
#     sys_prompt="",
#     model="gpt-3.5-turbo-1106",
#     temperature=0,
#     top_p=1,
#     json_output=False,
# ):
#     if model == "gpt-4":
#         model = "gpt-4-1106-preview"
#     sleep_time = 1
#     while not checker.can_execute_request():
#         print(
#             f"Network unstable, waiting...{sleep_time}s",
#             end="\r",
#             flush=True,
#         )
#         time.sleep(1)
#         sleep_time += 1
#     if not sys_prompt:
#         messages = [
#             {"role": "user", "content": prompt},
#         ]
#     else:
#         messages = [
#             {"role": "system", "content": sys_prompt},
#             {"role": "user", "content": prompt},
#         ]
#     retry_count = 0
#     max_retries = 20
#     interval_time = 5 if retry_count < 5 else 120
#     # print(messages)
#     while retry_count < max_retries:  # 限制重试次数
#         try:
#             # 使用stream=True来启用流式输出
#             start_time = time.time()
#             enc = tiktoken.encoding_for_model(model)

#             @timeout(60)
#             def get_response(stop_event=None):
#                 response = openai.chat.completions.create(
#                     model=model,
#                     messages=messages,
#                     temperature=temperature,  # this is the degree of randomness of the model's output
#                     top_p=top_p,
#                     stream=True,
#                     response_format={"type": "json_object"} if json_output else None,
#                 )
#                 return response

#             response = get_response()

#             # print(response.choices[0].message.content)
#             collected_chunks = []
#             collected_messages = []
#             for chunk in response:
#                 # chunk_time = (
#                 #     time.time() - start_time
#                 # )  # calculate the time delay of the chunk
#                 collected_chunks.append(chunk)  # save the event response
#                 chunk_message = chunk.choices[0].delta.content  # extract the message
#                 collected_messages.append(chunk_message)  # save the message
#                 print(
#                     f"{chunk_message}", end="", flush=True
#                 )  # print the delay and text
#             collected_messages = [m for m in collected_messages if m is not None]
#             full_reply_content = "".join(
#                 [m for m in collected_messages]
#             )  # response.choices[0].message.content
#             count_input_token = len(enc.encode(prompt + " " + sys_prompt))
#             count_output_token = len(enc.encode(full_reply_content))
#             # print("token:" + str(count_input_token + count_output_token))
#             cost_calculator.calculate(model, count_input_token, count_output_token)

#             return full_reply_content
#         except Exception as e:
#             retry_count += 1
#             print(f"An error occurred: {e}.")
#             for i in range(interval_time):
#                 print(
#                     f"Retrying in {interval_time-i} seconds...",
#                     end="\r",
#                     flush=True,
#                 )
#                 time.sleep(1)  # 暂停
#             print("Retrying ...                              ", end="\r", flush=True)

#     raise Exception("Max retries reached. Exiting without completion.")
MAX_RETRY = 5


# def openai_completion(
#     inputs,
#     sys_prompt="",
#     observe_window=None,
#     console_slience=False,
#     json_output=False,
#     top_p=1,
#     temperature=0,
#     model="gpt-3.5-turbo-1106",
# ):
#     """
#     发送至chatGPT，等待回复，一次性完成，不显示中间过程。但内部用stream的方法避免中途网线被掐。
#     inputs：
#         是本次问询的输入
#     sys_prompt:
#         系统静默prompt
#     llm_kwargs：
#         chatGPT的内部调优参数
#     history：
#         是之前的对话列表
#     observe_window = None：
#         用于负责跨越线程传递已经输出的部分，大部分时候仅仅为了fancy的视觉效果，留空即可。observe_window[0]：观测窗。observe_window[1]：看门狗
#     """
#     watch_dog_patience = 5  # 看门狗的耐心, 设置5秒即可
#     # headers, payload = generate_payload(
#     #     inputs, llm_kwargs, history, system_prompt=sys_prompt, stream=True
#     # )
#     messages = [
#         {"role": "user", "content": inputs},
#     ]
#     if sys_prompt:
#         messages.insert(0, {"role": "system", "content": sys_prompt})
#     enc = tiktoken.encoding_for_model(model)
#     retry = 0
#     while True:
#         try:
#             # make a POST request to the API endpoint, stream=False

#             # response = requests.post(
#             #     endpoint,
#             #     headers=headers,
#             #     proxies=proxies,
#             #     json=payload,
#             #     stream=True,
#             #     timeout=TIMEOUT_SECONDS,
#             # )
#             response = openai.chat.completions.create(
#                 model=model,
#                 messages=messages,
#                 temperature=temperature,  # this is the degree of randomness of the model's output
#                 top_p=top_p,
#                 stream=True,
#                 response_format={"type": "json_object"} if json_output else None,
#             )
#             break
#         except requests.exceptions.ReadTimeout:
#             retry += 1
#             traceback.print_exc()
#             if retry > MAX_RETRY:
#                 raise TimeoutError
#             if MAX_RETRY != 0:
#                 print(f"请求超时，正在重试 ({retry}/{MAX_RETRY}) ……")
#     collected_chunks = []
#     collected_messages = []
#     for chunk in response:
#         # chunk_time = (
#         #     time.time() - start_time
#         # )  # calculate the time delay of the chunk
#         collected_chunks.append(chunk)  # save the event response
#         chunk_message = chunk.choices[0].delta.content  # extract the message
#         collected_messages.append(chunk_message)  # save the message
#         print(f"{chunk_message}", end="", flush=True)  # print the delay and text

#     collected_messages = [m for m in collected_messages if m is not None]
#     full_reply_content = "".join(
#         [m for m in collected_messages]
#     )  # response.choices[0].message.content
#     count_input_token = len(enc.encode(inputs + " " + sys_prompt))
#     count_output_token = len(enc.encode(full_reply_content))
#     # print("token:" + str(count_input_token + count_output_token))
#     cost_calculator.calculate(model, count_input_token, count_output_token)

#     return full_reply_content


# def glm_completion(prompt, sys_prompt="", model="chatglm2-6b"):
#     # print("starting")
#     openai.api_base = "http://192.168.11.103:8000/v1"
#     openai.api_key = "none"
#     if not sys_prompt:
#         messages = [
#             {"role": "user", "content": prompt},
#         ]
#     else:
#         messages = [
#             {"role": "system", "content": sys_prompt},
#             {"role": "user", "content": prompt},
#         ]
#     retry_count = 0
#     max_retries = 5

#     while retry_count < max_retries:  # 限制重试次数
#         try:
#             # 使用stream=True来启用流式输出
#             response = openai.ChatCompletion.create(
#                 model=model,
#                 messages=messages,
#                 temperature=0,  # this is the degree of randomness of the model's output
#                 # stream=True,
#             )
#             return response.choices[0].message["content"]
#         except Exception as e:
#             retry_count += 1
#             print(f"An error occurred: {e}.")
#             for i in range(5):
#                 print(
#                     f"Retrying in {5-i} seconds...",
#                     end="\r",
#                     flush=True,
#                 )
#                 time.sleep(2)  # 暂停
#             print("Retrying ...                              ", end="\r", flush=True)

#     print("Max retries reached. Exiting without completion.")
#     return None


def gpt2pair(tc, tl, model="gpt-3.5-turbo-16k"):
    w = "{"
    for i in tc:
        i = i.replace('"', "").strip()
        w += f'"{i}":"",'
    w = w[:-1] + "}"

    message = f"""
    返回将translates填入却值的字典，不可额外添加单词，不可替换单词。

    {w}

    translates: {tl}
    """

    # display(Markdown(message))

    w = get_completion(message, model=model)
    w = w.replace("\n", "").replace(',"":""', "").replace(',""', "")
    # display(Markdown(w))
    try:
        json.loads(w)
    except Exception:
        message = f"返回仅用字典的语法符号修复json字典的语法错误后的完整字典，不得对键或值进行任何修改: \n{w}"
        w = get_completion(message)
        # display(Markdown("修复: "+w))
    w = json.loads(w)
    w = {k: ("" if v is None or v == [] or v == {} else v) for k, v in w.items()}
    keys = list(w.keys())
    values = list(w.values())
    for idx, item in enumerate(values):
        if idx + 1 < len(values) and not values[idx + 1]:
            values.pop(idx + 1)
            keys[idx] += " " + keys[idx + 1]
    w = dict(zip(keys, values))
    w = json.dumps(w)
    return w


@unique
class Model(StrEnum):
    GPT_3_5_TURBO = "gpt-3.5-turbo-0125"


@dataclass
class Message:
    role: str = ""  # required.
    content: str = ""  # required.


APPLICATION_JSON = "application/json"

# https://www.whatismybrowser.com/guides/the-latest-user-agent/chrome
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"
_CHAT_API_URL = "https://api.openai.com/v1/chat/completions"


async def chat(
    messages: list[Message],
    model: Model = Model.GPT_3_5_TURBO,
    top_p: float = 0.8,  # [0, 1]
    timeout: int = 10,
    api_key: str = OPENAI_API_KEY,
    json_output: bool = False,
) -> dict:

    headers = {
        # "User-Agent": USER_AGENT,
        "Content-Type": APPLICATION_JSON,
        "Authorization": f"Bearer {api_key}",
    }

    body = {
        "messages": list(map(lambda m: asdict(m), messages)),
        "model": model.value,
        "top_p": top_p,
        "temperature": 0,
        "stream": True,
        "response_format": {"type": "json_object"} if json_output else None,
    }

    result = chat_complation(headers, body)

    # Automatically .aclose() if the response body is read to completion.
    return result


_encoding_for_chat = tiktoken.get_encoding("cl100k_base")


def count_tokens(messages: list[Message]) -> int:
    tokens_count = 0

    for message in messages:
        # Every message follows "<im_start>{role/name}\n{content}<im_end>\n".
        tokens_count += 4

        for key, value in asdict(message).items():
            tokens_count += len(_encoding_for_chat.encode(value))

            # If there's a "name", the "role" is omitted.
            if key == "name":
                # "role" is always required and always 1 token.
                tokens_count += -1

    # Every reply is primed with "<im_start>assistant".
    tokens_count += 2

    return tokens_count


@unique
class Role(StrEnum):
    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"


def build_message(role: Role, content: str) -> Message:
    return Message(role=role.value, content=content.strip())
