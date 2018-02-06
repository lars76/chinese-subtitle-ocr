import logging
import os

import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import cv2
import numpy as np
import yaml

from detection import Detection
from recognition import Recognition

FONTS = ["NotoSansCJK-Regular.ttc", "wqy-zenhei.ttc", "SourceHanSansCN-Regular.otf", "simsun.ttc"]
FONT_COLOR = "white"
RECTANGLE_COLOR = "green"


def load_font(font_name, size):
    try:
        font = ImageFont.truetype(font_name, size)
    except IOError:
        logging.warning("Font {} not found".format(font_name))
        return None

    return font


def draw_text(draw, font, font2, pos_x, pos_y, text, prob, cyk=True):
    start, stop = pos_x
    y_start, y_end = pos_y
    draw.rectangle([(start, y_start), (stop, y_end)], outline=RECTANGLE_COLOR)
    draw.rectangle([(start + 1, y_start + 1), (stop - 1, y_end - 1)], outline=RECTANGLE_COLOR)
    draw.rectangle([(start + 2, y_start + 2), (stop - 2, y_end - 2)], outline=RECTANGLE_COLOR)
    probability = str(int(prob * 100))
    if cyk:
        draw.text((start, y_start - (stop - start)), text, fill=FONT_COLOR, font=font)
        draw.text((start, y_start - 1.5 * (stop - start)), probability + "%", fill=FONT_COLOR, font=font2)
    else:
        logging.info("Detected character {} ({} %)".format(text, probability))


def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

    with open("config.yml", "r") as config_file:
        cfg = yaml.load(config_file)

    det_cfg = cfg["detection"]
    rec_cfg = cfg["recognition"]

    logging.basicConfig(format="%(asctime)s %(module)-12s %(levelname)-8s %(message)s", level=cfg["log_level"])

    logging.info("Starting detection")

    detection = Detection(det_cfg)

    found_frames = detection.detect_subtitle_region(cfg["video"])

    y_start, y_end = detection.get_subtitle_region()
    char_width = detection.get_char_width()
    char_dist = detection.get_char_dist()
    if char_width == 0 or char_dist == 0:
        logging.error("Char width is 0")
        return

    logging.info(
        "Found y pos ({}, {}), character width {}, character distance {}".format(y_start, y_end, char_width, char_dist))

    recognition = Recognition(rec_cfg["model"], rec_cfg["weights"], rec_cfg["dictionary"])

    cyk = True
    for index, f in enumerate(FONTS):
        font = load_font(f, char_width)
        font2 = load_font(f, char_width // 2)
    if font is None:
        logging.error("No CYK font found")
        cyk = False
    else:
        logging.info("Loaded font {}".format(FONTS[index]))

    for frame in found_frames:
        text = []
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        for char_region, start, stop in detection.detect_char_regions(frame[y_start:y_end, ]):
            res = recognition.recognize_character(char_region)
            text.append((start, stop, res[1], res[2]))

        for start, stop, char, prob in text:
            draw.rectangle([(start, y_start), (stop, y_end)], outline=RECTANGLE_COLOR)
            draw.rectangle([(start + 1, y_start + 1), (stop - 1, y_end - 1)], outline=RECTANGLE_COLOR)
            draw.rectangle([(start + 2, y_start + 2), (stop - 2, y_end - 2)], outline=RECTANGLE_COLOR)

            probability = str(int(prob * 100)) + "%"
            if cyk:
                draw.text((start, y_start - (stop - start)), char, fill=FONT_COLOR, font=font)
                draw.text((start, y_start - 1.5 * (stop - start)), probability, fill=FONT_COLOR, font=font2)
            else:
                logging.info("Detected character {} ({})".format(char, probability))

        cv2.imshow('image', np.array(img))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
