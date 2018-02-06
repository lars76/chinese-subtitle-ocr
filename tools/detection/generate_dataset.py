import csv
import glob
import xml.etree.ElementTree as ET

INPUT_FOLDER = "images"
OUTPUT_FILE = "output.csv"


def parse_xml():
    rows = []
    for xml_file in glob.glob(INPUT_FOLDER + "/*xml"):
        tree = ET.parse(xml_file)

        filename = tree.findtext("./filename")
        path = tree.findtext("./path")
        if path == filename:
            path = xml_file.replace("xml", "jpg")
        height = tree.findtext("./size/height")
        width = tree.findtext("./size/width")

        name = tree.findtext("./object/name")
        xmin = tree.findtext("./object/bndbox/xmin")
        xmax = tree.findtext("./object/bndbox/xmax")
        ymin = tree.findtext("./object/bndbox/ymin")
        ymax = tree.findtext("./object/bndbox/ymax")
        rows.append([filename, path, height, width, name, xmin, xmax, ymin, ymax])
    return rows


def main():
    with open(OUTPUT_FILE, "w") as out:
        writer = csv.writer(out, delimiter=",")
        rows = parse_xml()

        # normalize ymin, ymax
        videos = {}
        for row in rows:
            filename, path, height, width, name, xmin, xmax, ymin, ymax = row
            ymin = int(ymin)
            ymax = int(ymax)
            video = filename[:filename.rfind("_")]
            if not videos.get(video):
                videos[video] = (ymin, ymax)
            else:
                videos[video] = (max(ymin, videos[video][0]), min(ymax, videos[video][1]))

        for row in rows:
            filename, path, height, width, name, xmin, xmax, _, _ = row
            ymin, ymax = videos[filename[:filename.rfind("_")]]
            writer.writerow([filename, path, height, width, name, xmin, xmax, ymin, ymax])


if __name__ == "__main__":
    main()
