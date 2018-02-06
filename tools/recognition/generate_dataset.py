import glob
import os
import random
from os.path import basename

import cv2
from PIL import ImageFont, ImageDraw, Image, ImageFilter
from train_model import TRAIN_DATA_DIR, VALIDATION_DATA_DIR

VIDEO_FOLDER = "videos"
FONT_FOLDER = "fonts"
CHARACTER_FILE = "dictionary.txt"
FRAME_FOLDER = "frames"

MAX_FRAMES = 40000
MAX_IMAGES_PER_CHARACTER = 300
TRAIN_RATIO = 0.8
SIZE = 32

GRAY_SCALE = False

GAUSSIAN_BLUR = False
RADIUS_START = 0.0
RADIUS_END = 1.0

FONT_SIZES = [30, 28, 33]

SHIFT_X_RANGES = [(0, 3), (0, 6), (0, 0)]
SHIFT_Y_RANGES = [(-2, 3), (-2, 5), (0, 0)]

ADD_SHADOW = True
SHADOW_FONTS = ["NotoSansCJK-Bold.ttc", "NotoSerifCJK-Black.ttc"]
SHADOW_WIDTH = 1


def get_characters(character_file):
    characters = []
    with open(character_file, "r") as f:
        characters = f.read().split("\n")

    if len(set(characters)) != len(characters):
        print("Warning: found multiple times the same character")

        from collections import Counter
        print(Counter(characters).most_common())

    return characters


def get_all_files(folder):
    files = []
    for r, d, f in os.walk(folder):
        for k in f:
            files.append(os.path.join(r, k))

    return files


def generate_patches(frame_folder, video_folder, count):
    if not os.path.exists(FRAME_FOLDER):
        os.makedirs(FRAME_FOLDER)

    patches = []
    videos = glob.glob(VIDEO_FOLDER + "/*")
    i = 1
    for k, video in enumerate(videos):
        video_name = os.path.splitext(os.path.basename(video))[0]

        cap = cv2.VideoCapture(video)
        max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # make sure we are not at the opening credits
        cap.set(cv2.CAP_PROP_POS_FRAMES, max_frames // 4)

        j = 0
        pos_frame = 0
        frame = None
        while pos_frame < max_frames and j < count // len(videos):
            print("{}/{}".format(i + j, count), end="\r")
            _, frame = cap.read()

            rnd_pos_x = random.randint(0, width - SIZE)
            rnd_pos_y = random.randint(0, height // 2)

            image_patch = frame[rnd_pos_y:rnd_pos_y + SIZE, rnd_pos_x:rnd_pos_x + SIZE]

            filename = os.path.join(FRAME_FOLDER, '{:0>5}.jpg').format(i + j + 1)
            patches.append(filename)
            cv2.imwrite(filename, image_patch)

            j += 1

        i += j
    print()

    return patches


def main():
    fonts = get_all_files(FONT_FOLDER)
    characters = get_characters(CHARACTER_FILE)
    patches = generate_patches(FRAME_FOLDER, VIDEO_FOLDER, MAX_FRAMES)

    for j, character in enumerate(characters):
        print("{}/{}".format(j + 1, len(characters)))
        print(character)

        for i in range(0, MAX_IMAGES_PER_CHARACTER):
            print("{}/{}".format(i + 1, MAX_IMAGES_PER_CHARACTER), end="\r")
            img = Image.open(random.choice(patches))

            chosen_font = random.choice(fonts)
            font_size_index = random.randint(0, len(FONT_SIZES) - 1)
            font_size = FONT_SIZES[font_size_index]
            font = ImageFont.truetype(chosen_font, font_size)

            draw = ImageDraw.Draw(img)
            w, h = draw.textsize(character, font=font)

            # needed for chinese characters
            # https://stackoverflow.com/questions/43060479/how-to-get-the-font-pixel-height-using-pil-imagefont
            _, (_, offset_y) = font.font.getsize(character)

            y = (font_size - h - offset_y) / 2
            y_shift_range = SHIFT_Y_RANGES[font_size_index]
            y += random.randint(y_shift_range[0], y_shift_range[1])

            x = (font_size - w) / 2
            x_shift_range = SHIFT_X_RANGES[font_size_index]
            x += random.randint(x_shift_range[0], x_shift_range[1])

            if not character == ' ':
                if ADD_SHADOW and basename(chosen_font) in SHADOW_FONTS:
                    sh = SHADOW_WIDTH
                    draw.text((x - sh, y), character, font=font, fill="black")
                    draw.text((x + sh, y), character, font=font, fill="black")
                    draw.text((x, y - sh), character, font=font, fill="black")
                    draw.text((x, y + sh), character, font=font, fill="black")

                    draw.text((x - sh, y - sh), character, font=font, fill="black")
                    draw.text((x + sh, y - sh), character, font=font, fill="black")
                    draw.text((x - sh, y + sh), character, font=font, fill="black")
                    draw.text((x + sh, y + sh), character, font=font, fill="black")

                draw.text((x, y), character, font=font)

            if GAUSSIAN_BLUR:
                radius = random.uniform(RADIUS_START, RADIUS_END)
                fil = ImageFilter.GaussianBlur(radius=radius)
                img = img.filter(fil)

            filename = "{:0>3}.jpg".format(i)

            if i < int(MAX_IMAGES_PER_CHARACTER * TRAIN_RATIO):
                character_folder = os.path.join(TRAIN_DATA_DIR, "{:0>4}".format(j))
            else:
                character_folder = os.path.join(VALIDATION_DATA_DIR, "{:0>4}".format(j))
            if not os.path.exists(character_folder):
                os.makedirs(character_folder)

            if GRAY_SCALE:
                img = img.convert(mode="L")

            img.save(os.path.join(character_folder, filename))
        print()


if __name__ == "__main__":
    main()
