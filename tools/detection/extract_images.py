import glob
import os
import re
from datetime import datetime

import cv2

IMAGE_FOLDER = "images"
VIDEO_FOLDER = "videos"
MAX_IMAGES_PER_VIDEO = 100
SKIP_FRAMES = 100
SUBTITLE_CORRECTION = 5


def parse_subtitle(filename):
    if not os.path.isfile(filename):
        return []

    subtitle_time = []
    with open(filename, 'r') as f:
        reg = re.findall(r'(\d+?)\n(.*?) --> (.*?)\n(.*?)\n', f.read())

        for index, line in enumerate(reg):
            start = line[1].replace(',', '.')
            end = line[2].replace(',', '.')

            start_time = datetime.strptime(start, '%H:%M:%S.%f')
            end_time = datetime.strptime(end, '%H:%M:%S.%f')
            epoch = datetime.utcfromtimestamp(0)

            start_time = start_time.replace(year=1970)
            end_time = end_time.replace(year=1970)
            duration = end_time - start_time

            if duration.total_seconds() < 0:
                continue

            conv1 = (start_time - epoch).total_seconds() * 1000.0
            conv2 = (end_time - epoch).total_seconds() * 1000.0

            subtitle_time.append((conv1 + SUBTITLE_CORRECTION, conv2 - SUBTITLE_CORRECTION))

    return subtitle_time


def main():
    for video in glob.glob(VIDEO_FOLDER + "/*"):
        if ".srt" in video:
            continue
        print(video)
        video_name = os.path.splitext(os.path.basename(video))[0]

        cap = cv2.VideoCapture(video)
        max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max_frames // 4)

        srt_subtitle = parse_subtitle(os.path.join(VIDEO_FOLDER, video_name + ".srt"))
        if srt_subtitle:
            cap.set(cv2.CAP_PROP_POS_MSEC, srt_subtitle[0][0])

        pos_frame = 0
        i = 0
        pos_subtitle = 0
        while pos_frame < max_frames and i < MAX_IMAGES_PER_VIDEO:
            pos_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            print("{}/{}".format(i + 1, MAX_IMAGES_PER_VIDEO), end="\r")
            _, frame = cap.read()

            if pos_frame % SKIP_FRAMES > 0:
                continue

            if not srt_subtitle or (msec >= srt_subtitle[pos_subtitle][0] and msec <= srt_subtitle[pos_subtitle][1]):
                cv2.imwrite(os.path.join(IMAGE_FOLDER, "{}_{}.jpg").format(video_name, pos_frame), frame)
                i += 1
            elif msec > srt_subtitle[pos_subtitle][1]:
                pos_subtitle += 1
                if pos_subtitle >= len(srt_subtitle):
                    break
                cap.set(cv2.CAP_PROP_POS_MSEC, srt_subtitle[pos_subtitle][0])


if __name__ == "__main__":
    main()
