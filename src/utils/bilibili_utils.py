from bilibili_api import login, video_uploader, settings
import pickle, os, time, asyncio, sys, toml, re, httpx
from .status_utils import print_status
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys

from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from .captcha_utils import predict
import json
import base64
from .cache_utils import load_cache, save_cache


def get_location(element, browser):
    # 获取元素在屏幕上的位置信息
    location = element.location
    # size = element.size
    # height = size["height"]
    # width = size["width"]
    # left = location["x"]
    # top = location["y"]
    # right = left + width
    # bottom = top + height
    # script = f"return {{'left': {left}, 'top': {top}, 'right': {right}, 'bottom': {bottom}}};"
    # rect = browser.execute_script(script)

    # # # 计算元素的中心坐标
    # # center_x = int((rect['left'] + rect['right']) / 2)
    # # center_y = int((rect['top'] + rect['bottom']) / 2)
    # # # 计算元素左上
    # center_x = int(rect["left"])
    # center_y = int(rect["top"])
    x = location.get("x")
    y = location.get("y")
    return x, y


def get_credential():
    """
    尝试从缓存中获取Bilibili登录凭证。
    如果缓存不存在或凭证无效，则通过扫描二维码的方式重新登录并更新缓存。

    Returns:
        Credential: Bilibili API登录凭证。
    """
    credential_path = "asset/cookie/credential.pkl"

    def login_and_save_credential():
        """
        登录Bilibili并保存新的凭证到文件。
        """
        try:
            credential = login.login_with_qrcode_term()
            credential.raise_for_no_bili_jct()
            credential.raise_for_no_sessdata()
            with open(credential_path, "wb") as f:
                pickle.dump(credential, f)
            return credential
        except Exception as e:
            print(f"登录失败: {e}")
            raise

    # 尝试从文件加载凭证，如果文件不存在或加载失败，则重新登录
    if os.path.exists(credential_path):
        with open(credential_path, "rb") as f:
            try:
                credential = pickle.load(f)
                credential.raise_for_no_bili_jct()
                credential.raise_for_no_sessdata()
            except Exception as e:
                print(f"加载凭证失败或凭证无效: {e}")
                credential = login_and_save_credential()
    else:
        credential = login_and_save_credential()

    return credential


async def deliver(video, credential):
    # sys.exit()
    try:
        meta = {
            "act_reserve_create": 0,
            "copyright": 1,
            "source": "",
            "desc": video["Description"],
            "desc_format_id": 0,
            "dynamic": "",
            "interactive": 0,
            "no_reprint": 0,
            "open_elec": 0,
            "origin_state": 0,
            "subtitles": {"lan": "", "open": 0},
            "tag": video["Tag"],
            "tid": video["Tid"],
            "title": video["Title"],
            "up_close_danmaku": False,
            "up_close_reply": False,
            "up_selection_reply": False,
            "dtime": 0,
        }
        page = video_uploader.VideoUploaderPage(
            path=video["Video"],
            title=video["Title"],
            description=video["Description"],
        )

        uploader = video_uploader.VideoUploader(
            [page], meta, credential, cover=video["Thumbnail"]
        )
        if len(video["Title"]) > 80:
            raise ValueError(f"Title too long (large than 80): {video['Title']}")

        # print(f"Delivering {video['Title']}")

        flag = [True]

        msg = [f"Delivering {video['Title']}"]
        print_status(flag, msg)

        @uploader.on("__ALL__")
        async def ev(data):
            nonlocal msg
            msg[0] = f"Deliver {video['Title']}: {data['name']}"

            # print(data)

        await uploader.start()
        flag[0] = False
    except ValueError as e:
        print(f"Deliver {video['Title']} failed: {e}")
        sys.exit()

    except Exception as e:
        flag[0] = False
        raise e


def upload(path_to_file, img_path, tags, desc, title, season, metadatapath):
    # 创建一个 WebDriver 实例，指定使用的浏览器
    path_to_file = os.path.abspath(path_to_file)
    img_path = os.path.abspath(img_path)

    chrome_binary_path = os.path.expanduser(
        "~/delta_context/asset/chrome-linux64/chrome"
    )
    options = Options()
    options.set_capability("goog:loggingPrefs", {"performance": "ALL"})
    options.binary_location = chrome_binary_path
    # options.add_argument("--verbose")
    options.add_argument("--no-sandbox")
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")  # Sometimes necessary for headless mode
    # Bypass OS security model, required on Linux
    options.add_argument("window-size=1920,1080")
    options.add_argument(
        "--disable-dev-shm-usage"
    )  # Overcome limited resource problems

    driver = webdriver.Chrome(options=options)
    # driver.maximize_window()
    # 打开一个页面
    driver.get("https://www.bilibili.com")

    cookies = toml.load("asset/cookie/cookies.toml")["cookies"]

    for cookie in cookies:
        driver.add_cookie(cookie)

    driver.get(
        "https://member.bilibili.com/platform/upload/video/frame?page_from=creative_home_top_upload"
    )
    wait = WebDriverWait(driver, 10)

    # 等待直到文件输入元素变得可用
    wait.until(
        EC.presence_of_element_located((By.XPATH, "//div[@class='bcc-upload-wrapper']"))
    )
    driver.find_element(
        By.XPATH, "//input[@type='file' and contains(@accept, 'mp4')]"
    ).send_keys(path_to_file)

    # driver.find_element(
    #     By.XPATH, "//button[@class='bcc-button vp-nd-f bcc-button--primary small']"
    # ).click()

    wait.until(
        EC.presence_of_element_located((By.XPATH, "//div[@class='cover-upload-btn']"))
    )
    driver.find_element(
        By.XPATH, "//input[@type='file' and contains(@accept, 'image/png')]"
    ).send_keys(img_path)

    wait.until(
        EC.presence_of_element_located(
            (
                By.XPATH,
                "//div[@class='cover-select-footer-pick']",
            )
        )
    )
    time.sleep(0.1)

    driver.find_element(
        By.XPATH,
        "//button[@class='bcc-button bcc-button--primary large']//span[contains(text(), '完成')]",
    ).click()
    time.sleep(0.25)

    wait.until(
        EC.presence_of_element_located(
            (
                By.XPATH,
                "//input[@placeholder='清晰明了表明内容亮点的标题会更受观众欢迎哟！']",
            )
        )
    )
    title_elem = driver.find_element(
        By.XPATH, "//input[@placeholder='清晰明了表明内容亮点的标题会更受观众欢迎哟！']"
    )
    title_elem.clear()
    # title_elem.send_keys(title)
    JS_ADD_TEXT_TO_INPUT = """
  var elm = arguments[0], txt = arguments[1];
  elm.value += txt;
  elm.dispatchEvent(new Event('change'));
  """
    driver.execute_script(JS_ADD_TEXT_TO_INPUT, title_elem, title)

    time.sleep(1)
    print("send tags", tags)
    tag_failed_count = 0
    for tag in tags:
        tag = tag.strip()
        wait.until(
            EC.presence_of_element_located(
                (
                    By.XPATH,
                    "//input[@placeholder='按回车键Enter创建标签']",
                )
            )
        )
        tag_input = driver.find_element(
            By.XPATH, "//input[@placeholder='按回车键Enter创建标签']"
        )
        tag_input.click()

        tag_input.send_keys(tag)

        tag_input.send_keys(Keys.ENTER)
        # time.sleep(1)

        try:
            wait.until(
                EC.presence_of_element_located(
                    (
                        By.XPATH,
                        f"//div[@class='tag-pre-wrp']//p[contains(text(), '{tag}')]",
                    )
                )
            )
            print("tag", tag)
        except Exception:
            # print("tag retry")
            # wait.until(
            #     EC.presence_of_element_located(
            #         (
            #             By.XPATH,
            #             "//input[@placeholder='按回车键Enter创建标签']",
            #         )
            #     )
            # )
            # tag_input = driver.find_element(
            #     By.XPATH, "//input[@placeholder='按回车键Enter创建标签']"
            # )
            # tag_input.click()

            # tag_input.send_keys(tag)

            # tag_input.send_keys(Keys.ENTER)
            # wait.until(
            #     EC.presence_of_element_located(
            #         (
            #             By.XPATH,
            #             f"//div[@class='tag-pre-wrp']//p[contains(text(), '{tag}')]",
            #         )
            #     )
            # )

            tag_failed_count += 1
    if tag_failed_count == len(tags):
        # print("tag fill failed")
        raise ValueError("tag fill failed")
    time.sleep(0.1)
    desc_elem = driver.find_element(
        By.XPATH,
        "//div[contains(@data-placeholder, '填写更全面的相关信息，让更多的人能找到你的视频吧')]",
    )
    desc_paragraphs = desc.split("\n")
    for idx, paragraph in enumerate(desc_paragraphs):
        if idx == 0:
            p_element = desc_elem.find_element(By.XPATH, ".//p")
            driver.execute_script(
                "arguments[0].innerText = arguments[1];", p_element, paragraph
            )

        else:
            # 创建一个新的 <p> 元素并设置其文本
            script = """
            var para = document.createElement("p");
            para.innerText = arguments[1];
            arguments[0].appendChild(para);
            """
            driver.execute_script(script, desc_elem, paragraph)

    if season:
        wait.until(
            EC.presence_of_element_located(
                (
                    By.XPATH,
                    "//div[@class='form-item']//div[@class='season-enter']",
                )
            )
        )
        driver.find_element(
            By.XPATH, "//div[@class='form-item']//div[@class='season-enter']"
        ).click()

        # 首先，定位到包含所有 season-item 的父元素
        season_list = driver.find_element(By.CLASS_NAME, "season-list.season-dropdown")

        # 然后，找到这个元素下所有的 season-item 元素
        season_items = season_list.find_elements(By.CLASS_NAME, "season-item")

        # 遍历这些元素，并获取它们的文本
        season_texts = [item.text for item in season_items]

        # season_texts 现在包含了所有 season-item 元素的文本
        print(season_texts)
        season = season.strip().replace(" ", "_")
        print(season)

        if season in season_texts:
            season_items[season_texts.index(season)].click()
        else:
            driver.find_element(By.CLASS_NAME, "season-add").click()
            wait.until(
                EC.presence_of_element_located(
                    (
                        By.XPATH,
                        "//input[contains(@placeholder, '请输入标题，创建合集')]",
                    )
                )
            )
            season_add_elem = driver.find_element(
                By.XPATH, "//input[contains(@placeholder, '请输入标题，创建合集')]"
            )
            season_add_elem.click()
            season_add_elem.send_keys(season)
            driver.find_element(
                By.XPATH, "//button[.//span[text()='创建并加入']]"
            ).click()

        wait.until(
            EC.presence_of_element_located(
                (
                    By.XPATH,
                    f"//span[@class='season-enter-text']//span[contains(text(), '{season}')]",
                )
            )
        )
        driver.find_element(
            By.XPATH,
            "//span[@class='bcc-checkbox-label' and contains(text(), '此稿件不生成更新推送')]",
        ).click()
        wait.until(
            EC.presence_of_element_located(
                (
                    By.XPATH,
                    "//div[@class='video-season-checkbox-item']//label[@class='bcc-checkbox bcc-checkbox-checked']",
                )
            )
        )
    # time.sleep(1200)
    driver.find_element(By.XPATH, "//span[@class='v-popover-close-wrp']").click()
    more_setting_elem = driver.find_element(
        By.XPATH, "//span[@class='label' and contains(text(), '更多设置')]"
    )
    # actions = ActionChains(driver)
    # actions.move_to_element(more_setting_elem).perform()
    wait.until(EC.element_to_be_clickable(more_setting_elem))
    more_setting_elem.click()

    wait.until(
        EC.presence_of_element_located(
            (
                By.XPATH,
                "//div[@class='setting']",
            )
        )
    )
    driver.find_element(
        By.XPATH,
        "//label[@class='bcc-checkbox bcc-checkbox-checked']//span[@class='bcc-checkbox-label' and contains(text(), '未经作者授权 禁止转载')]",
    ).click()
    wait.until(
        EC.presence_of_element_located(
            (
                By.XPATH,
                "//label[@class='bcc-checkbox']//span[@class='bcc-checkbox-label' and contains(text(), '未经作者授权 禁止转载')]",
            )
        )
    )

    WebDriverWait(driver, 600).until(
        EC.presence_of_element_located(
            (
                By.XPATH,
                "//span[@class='success' and contains(text(), '上传完成')]",
            )
        )
    )
    driver.find_element(
        By.XPATH,
        "//span[@class='submit-add' and contains(text(), '立即投稿')]",
    ).click()
    # for entry in driver.get_log("browser"):
    #     print("browser", entry)
    time.sleep(1)

    try:
        wait.until(
            EC.presence_of_element_located(
                (
                    By.XPATH,
                    "//div[@class='step-des' and contains(text(), '稿件投递成功')]",
                )
            )
        )
    except Exception:
        xpath = '//*[@class="geetest_item_wrap"]'
        captcha_elem = wait.until(EC.presence_of_element_located((By.XPATH, xpath)))
        f = captcha_elem.get_attribute("style")
        captcha_url = re.findall('url\("(.+?)"\);', f)
        captcha_url = captcha_url[0] if captcha_url else None
        if not captcha_url:
            raise ValueError("captcha url not found")
        img_bytes = httpx.get(captcha_url).content
        img_base64_str = base64.b64encode(img_bytes).decode("utf-8")
        url = "http://127.0.0.1:5000/detect"
        data = {"image_base64": img_base64_str}
        response = httpx.post(url, json=data)
        detection_list = response.json()
        if "error" in detection_list:
            raise ValueError(detection_list["error"])
        print("response", response)
        print("detection_list", detection_list)
        # 送入模型识别
        plan = predict(img_bytes, detection_list)
        print("plan", plan)
        # 获取验证码坐标
        element = driver.find_element(By.CLASS_NAME, "geetest_item_wrap")
        # X, Y = get_location(element, driver)
        X, Y = element.location["x"], element.location["y"]
        scroll_y = driver.execute_script("return window.pageYOffset;")
        print("scroll_y", scroll_y)
        Y -= scroll_y
        print("captcha_elem", X, Y)
        # 前端展示对于原图的缩放比例
        lan_x = 306 / 334
        lan_y = 343 / 384
        for i, crop in enumerate(plan):
            x1, y1, x2, y2 = crop
            x, y = [(x1 + x2) / 2, (y1 + y2) / 2]
            print(X + x * lan_x, Y + y * lan_y)

            ActionChains(driver).move_by_offset(
                X + x * lan_x, Y + y * lan_y
            ).click().perform()
            ActionChains(driver).move_by_offset(
                -(X + x * lan_x), -(Y + y * lan_y)
            ).perform()  # 将鼠标位置恢复到移动前
            time.sleep(0.5)
        driver.save_screenshot("screenshot.png")
        xpath = '//*[@class="geetest_commit_tip"]'
        wait.until(EC.presence_of_element_located((By.XPATH, xpath))).click()

        time.sleep(1)
        # xpath = "/html/body/div[4]/div[2]/div[6]/div/div/div[3]/div/a[2]"
        # wait.until(EC.presence_of_element_located((By.XPATH, xpath))).click()
        driver.save_screenshot("screenshot1.png")
        WebDriverWait(driver, 60).until(
            EC.presence_of_element_located(
                (
                    By.XPATH,
                    "//div[@class='step-des' and contains(text(), '稿件投递成功')]",
                )
            )
        )
    info = load_cache(metadatapath)
    info["is_delivered"] = True
    save_cache(info, metadatapath)

    logs = driver.get_log("performance")
    with open("bvid_msg_log.txt", "w") as f:
        f.write(str(logs))
    bvid_msgs = []
    for log in logs:
        if "bvid" in log["message"] and "aid" in log["message"]:
            # print(json.loads(log["message"])["message"])
            try:
                bvid_msg = json.loads(log["message"])["message"]["params"]["request"][
                    "postData"
                ]
                bvid_msgs.append(bvid_msg)
            except Exception:
                continue

    if not bvid_msgs:
        # print(logs)
        raise Exception("bvid not found")
    # print(bvid_msg)

    reg = r"\|\{.*?\}\|"
    matches = []
    for bvid_msg in bvid_msgs:
        match = re.findall(reg, bvid_msg)
        if match:
            matches.append(match[0])
    # matches = matches[0]
    # print(matches)
    res_id = None
    for match in matches:
        match = match.replace("|", "")
        match = json.loads(match)
        try:
            res_id = match["value"]["res"]
            break
        except Exception:
            continue

    print("done aid and bvid", res_id)
    driver.quit()
    # time.sleep(1200)
    return res_id
