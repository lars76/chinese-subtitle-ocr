import csv
import os
from multiprocessing.dummy import Pool
from urllib import request

# dataset can be found here: https://github.com/openimages/dataset
# https://storage.googleapis.com/openimages/2017_11/images_2017_11.tar.gz

IMAGES = 10000
PATH = "images/"


def download(url):
    filename = url[url.rfind("/") + 1:]
    if os.path.isfile(os.path.join(PATH, filename)):
        print("File {} already exists".format(filename))
        return
    print("Downloading {}".format(filename))
    try:
        data = request.urlopen(url).read()
        with open(os.path.join(PATH, filename), "wb") as f:
            f.write(data)
        print("Downloaded {}".format(filename))
    except:
        print("Error {}".format(filename))


with open("images.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')

    urls = []
    i = 0
    for row in reader:
        if i >= IMAGES:
            break
        url = row[2]
        if not "http" in url:
            continue
        urls.append(url)
        i += 1

    pool = Pool()
    pool.map(download, urls)
    pool.close()
    pool.join()
