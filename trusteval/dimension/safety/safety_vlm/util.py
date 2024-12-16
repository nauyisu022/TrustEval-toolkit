import os
import base64
import requests
from PIL import Image, ImageFont, ImageDraw

def encode_image(image_path):
    if image_path.startswith("http") or image_path.startswith("https"):
        response = requests.get(image_path)
        return base64.b64encode(response.content).decode('utf-8')
    else:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


def typo_format_text(text, font_size=60, max_width=1024):
    try:
        font = ImageFont.truetype('arial.ttf', font_size)
    except IOError:
        try:
            font = ImageFont.truetype('DejaVuSans.ttf', font_size)
        except IOError:
            print("Could not find specified fonts. Loading default.")
            font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', font_size)
    img = Image.new('RGB', (max_width, 100), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    word_list = text.split(" ")
    word_num = len(word_list)
    formated_text = word_list[0]
    cur_line_len = draw.textlength(formated_text, font=font)
    line_num = 1
    for i in range(1, word_num):
        cur_line_len += draw.textlength(" "+word_list[i], font=font)
        if cur_line_len < max_width:
            formated_text += " "+word_list[i]
        else:
            formated_text += "\n "+word_list[i]
            cur_line_len= draw.textlength(" "+word_list[i], font=font)
            line_num += 1
    return formated_text, line_num

def typo_draw_img(formated_text, line_num, font_size=60, max_width=1024):
    try:
        font = ImageFont.truetype('arial.ttf', font_size)
    except IOError:
        try:
            font = ImageFont.truetype('DejaVuSans.ttf', font_size)
        except IOError:
            print("Could not find specified fonts. Loading default.")
            font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', font_size)
    max_height = font_size * (line_num + 1)
    img = Image.new('RGB', (max_width, max_height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, font_size/2.0), formated_text, (0, 0, 0), font=font)
    return img

