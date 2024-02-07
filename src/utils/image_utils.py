from PIL import Image


def resize_img(img_path):
    img = Image.open(img_path)
    img_resized = img.resize((1920, 1080), Image.BICUBIC)
    img_resized.save(img_path)
