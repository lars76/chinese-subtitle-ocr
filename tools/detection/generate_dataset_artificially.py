import csv
import os
import random

import numpy as np
from PIL import ImageFont, ImageDraw, Image, ImageFilter, ImageEnhance

IMAGE_SIZE = (300, 300)
ITERATIONS = 1

OUTPUT_FILE = "output.csv"
OUTPUT_FOLDER = "images_output"
IMAGE_FOLDER = "images"
RESIZE = True
FILTER = Image.NEAREST
KEEP_ASPECT_RATIO = False
RANDOM_COLOR = False
GAUSSIAN_BLUR = True
RADIUS = 1.0
ADJUST_BRIGHTNESS = 0.7

FONT_FOLDER = "fonts"
START_FONT_SIZE = 300
MIN_TEXT_LENGTH = 5
MAX_TEXT_LENGTH = 15


def get_all_files(folder):
    files = []
    for r, d, f in os.walk(folder):
        for k in f:
            files.append(os.path.join(r, k))

    return files


def new_pos(w, h, img_width, img_height):
    return (w / img_width) * IMAGE_SIZE[0], (h / img_height) * IMAGE_SIZE[1]


def main():
    images = get_all_files(IMAGE_FOLDER)
    fonts = get_all_files(FONT_FOLDER)

    characters = []
    for code_point in range(0x4e00, 0x9fff + 1):
        characters.append(chr(code_point))

    # some characters aren't properly converted
    characters = characters[:-42]

    max_text = ''.join(characters[:MIN_TEXT_LENGTH])

    out = open(OUTPUT_FILE, "w")
    writer = csv.writer(out, delimiter=',')

    for j in range(0, ITERATIONS):
        print("Iteration {}/{}".format(j + 1, ITERATIONS))
        for i, elem in enumerate(images):
            print("{}/{}".format(i + 1, len(images)), end="\r")

            img = Image.open(elem)

            img_width, img_height = img.size

            if ADJUST_BRIGHTNESS < 1.0:
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(ADJUST_BRIGHTNESS)

            font_size = START_FONT_SIZE

            # too slow, improve this
            size = img_width
            while size >= img_width - img_width // 10:
                font = ImageFont.truetype(random.choice(fonts), font_size)
                size = font.getsize(max_text)[0]
                font_size -= 5

            text_length = random.randint(MIN_TEXT_LENGTH, MAX_TEXT_LENGTH)
            text = ""
            for _ in range(0, text_length):
                text += random.choice(characters)
            if i % 2 == 0 and text_length - 1 > 6:
                k = random.randint(1, text_length - 1)
                text = text[:k - 1] + " " + text[k + 1:]

            text_width, text_height = font.getsize(text)

            while img_width - text_width - img_width // 10 < img_width // 10:
                text = text[:-1]
                text_width, text_height = font.getsize(text)

            x = random.randint(img_width // 10, img_width - text_width - img_width // 10)
            y = random.randint(img_height // 10, img_height - text_height - img_height // 10)

            color = random.randint(0, 0xFFFFFF)
            if not RANDOM_COLOR or i % 2 == 0:
                color = "white"

            draw = ImageDraw.Draw(img)
            draw.text((x, y), text, font=font, fill=color)

            if GAUSSIAN_BLUR:
                radius = random.randint(1, 10)
                fil = ImageFilter.GaussianBlur(radius=RADIUS / radius)
                img = img.filter(fil)

            _, (_, offset_y) = font.font.getsize(text)
            y += offset_y // 2

            start = (x, y)
            end = (x + text_width, y + text_height)

            if RESIZE:
                if KEEP_ASPECT_RATIO:
                    img.thumbnail(IMAGE_SIZE)

                    img2 = Image.new("RGB", IMAGE_SIZE)
                    img2.paste(img, ((IMAGE_SIZE[0] - img.size[0]) // 2,
                                     (IMAGE_SIZE[1] - img.size[1]) // 2))
                    img = img2
                else:
                    img = img.resize(IMAGE_SIZE, FILTER)
                    draw = ImageDraw.Draw(img)
                img_width, img_height = img.size

                text = new_pos(text_width, text_height, img_width, img_height)
                start = new_pos(x, y, img_width, img_height)
                end = tuple(map(sum, zip(start, text)))

            filename = "{}_{}.jpg".format(j, i)
            path = os.path.join(OUTPUT_FOLDER, filename)

            img.save(path, format="JPEG")
            writer.writerow(
                [filename, path, img_height, img_width, "subtitle", int(start[0]), int(end[0]), int(start[1]),
                 int(end[1])])

        np.random.shuffle(images)

    out.close()


if __name__ == "__main__":
    main()
