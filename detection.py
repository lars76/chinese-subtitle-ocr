import logging
from itertools import tee

import cv2
import numpy as np
import tensorflow as tf


class Detection:
    def __load_graph(self, graph):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph, "rb") as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        return detection_graph

    def __set_tensors(self):
        detection_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = self.graph.get_tensor_by_name('detection_scores:0')
        detection_classes = self.graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.graph.get_tensor_by_name('num_detections:0')

        self.detection_tensors = [detection_boxes, detection_scores, detection_classes, num_detections]
        self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')

    def __add_dict(self, old_dict, y_min, y_max, dist=10):
        found = False
        new_dict = {}
        for k, v in old_dict.items():
            if np.abs(y_min - k[0]) <= dist and np.abs(y_max - k[1]) <= dist:
                new_pos = (max(y_min, k[0]), min(y_max, k[1]))
                new_dict[new_pos] = old_dict[k] + 1
                found = True
            else:
                new_dict[k] = old_dict[k]

        if not found:
            self.logger.debug("Adding new region ({}, {})".format(y_min, y_max))
            old_dict[(y_min, y_max)] = 1
            return old_dict

        return new_dict

    def __init__(self, cfg):
        self.logger = logging.getLogger(__name__)
        self.graph = self.__load_graph(cfg["frozen_graph"])
        self.logger.debug("Loaded graph")
        if self.graph:
            self.__set_tensors()
            self.logger.debug("Set tensors")

        self.batch_size = cfg["batch_size"]
        self.threshold = cfg["threshold"]
        self.min_box_matches = cfg["min_box_matches"]

        self.bilateral_filter = cfg["bilateral_filter"]
        self.diameter = cfg["diameter"]
        self.sigma_color = cfg["sigma_color"]
        self.sigma_space = cfg["sigma_space"]

        self.gaussian_blur = cfg["gaussian_blur"]
        self.kernel_size = cfg["kernel_size"]
        self.standard_deviation = cfg["standard_deviation"]

        self.block_size = cfg["block_size"]
        self.constant = cfg["constant"]

        self.char_min_coeff = cfg["char_min_coeff"]
        self.char_max_coeff = cfg["char_max_coeff"]
        self.char_min_dist = cfg["char_min_dist"]

        self.grp_min_coeff = cfg["grp_min_coeff"]
        self.grp_min_dist_coeff = cfg["grp_min_dist_coeff"]

        self.char_width = 0
        self.char_dist = 0
        self.subtitle_region = (0, 0)

    def __pairwise(self, iterable):
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)

    def __calculate_char_width(self, image):
        regions = self.__calculate_regions(image, image.shape[0] * self.char_min_coeff,
                                           image.shape[0] * self.char_max_coeff, self.char_min_dist)
        char_width = 0
        if regions:
            char_width = int(np.max([region[2] for region in regions]))

        char_dist = 0
        # char distance always requires at least two characters
        if len(regions) >= 2:
            char_dist = int(np.min([np.abs(region2[0] - region1[1]) for region1, region2 in self.__pairwise(regions)]))

        return char_width, char_dist

    def detect_char_regions(self, image):
        regions = self.__calculate_regions(image, self.char_width * self.grp_min_coeff, image.shape[1],
                                           self.char_width * self.grp_min_dist_coeff)
        for start, stop, dist in regions:
            for window in range(start, stop, self.char_width + self.char_dist):
                yield (image[:, window: self.char_width + window], window, self.char_width + window)

    def __calculate_regions(self, image, min_width, max_width, min_dist, backwards=10):
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = None
        if self.bilateral_filter:
            blurred = cv2.bilateralFilter(grayscale, self.diameter, self.sigma_color, self.sigma_space)
        if self.gaussian_blur:
            blurred = cv2.GaussianBlur(blurred, (self.kernel_size, self.kernel_size), self.standard_deviation)
        if blurred is None:
            blurred = grayscale.copy()
        threshold_img = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                              self.block_size, self.constant)

        result = []
        start = 0
        width = 0
        for index in range(0, threshold_img.shape[1] - 1):
            col = threshold_img[:, index]
            zeros = np.count_nonzero(col == 0)

            if (zeros > 0 and width == 0) or width >= max_width:
                start = index
                width = 1
            elif width > 0 and zeros == 0:
                if len(result) > 0 and np.abs(result[-1][1] - start) <= min_dist:
                    result[-1][1] = index
                elif width >= min_width:
                    result.append([start, index, width])
                elif width > backwards and np.count_nonzero(
                        threshold_img[:, index - backwards:index] == 0) >= backwards:
                    continue
                width = 0
            elif width > 0:
                width += 1

        if self.logger.getEffectiveLevel() == logging.DEBUG:
            img_copy = image.copy()
            for index, region in enumerate(result):
                cv2.rectangle(img_copy, (region[0], 3), (region[1], region[2]), (0, 255, 0), 1)

            three_channels = cv2.cvtColor(threshold_img, cv2.COLOR_GRAY2BGR)
            side_by_side = np.concatenate((img_copy, three_channels), axis=0)
            cv2.imshow("Image", side_by_side)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return result

    def get_char_dist(self):
        return self.char_dist

    def get_char_width(self):
        return self.char_width

    def get_subtitle_region(self):
        return self.subtitle_region

    def detect_subtitle_region(self, video, offset=0, step=80):
        cap = cv2.VideoCapture(video)
        self.logger.debug("Loaded video")

        max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames == 0:
            self.logger.error("Could not load video {}".format(video))
            return

        self.logger.debug("max_frames: {}".format(max_frames))
        if offset > max_frames:
            self.logger.warning("Offset > max_frames ({} > {})".format(offset, max_frames))
            offset = 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, offset)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.logger.debug("video width: {}, video height: {}".format(width, height))

        frames = np.empty((self.batch_size, height, width, 3))

        if offset == 0:
            offsets = np.arange(start=max_frames * 1 / 4, stop=max_frames * 3 / 4, step=step)
        else:
            offsets = np.arange(start=offset, stop=max_frames * 3 / 4, step=step)

        np.random.shuffle(offsets)

        sess = tf.Session(graph=self.graph)
        self.logger.debug("Started session")

        batch = 0
        regions = {}
        found_frames = []
        char_widths = []
        char_dists = []
        while True:
            if regions:
                subtitle_region = max(regions, key=regions.get)
                self.logger.debug("Found region {} ({} occurrences)".format(subtitle_region, regions[subtitle_region]))
                self.logger.info(
                    "{}/{}".format(min(regions[subtitle_region], self.min_box_matches), self.min_box_matches))

                if regions[subtitle_region] >= self.min_box_matches:
                    sess.close()
                    cap.release()
                    self.logger.debug("Stopped session")
                    break
                self.logger.debug("Matches {}".format(regions))

            frames2 = []
            for index in range(0, self.batch_size):
                offset = offsets[index + batch * self.batch_size]
                cap.set(cv2.CAP_PROP_POS_FRAMES, offset)
                _, frame = cap.read()
                frames[index] = frame
                frames2.append(frame)

            (boxes, scores, classes, num) = sess.run(self.detection_tensors, feed_dict={self.image_tensor: frames})
            for i in range(0, scores.shape[0]):
                pos = i + batch * self.batch_size

                maximal = np.argmax(scores[i])
                self.logger.debug(
                    "Frame {} (probability {} %)".format(offsets[pos], (scores[i][maximal] * 100).astype(int)))
                if scores[i][maximal] <= self.threshold:
                    continue

                box = tuple(boxes[i][maximal].tolist())

                y_min = int(box[0] * height)
                x_min = int(box[1] * width)

                y_max = int(box[2] * height)
                x_max = int(box[3] * width)

                regions = self.__add_dict(regions, y_min, y_max)

                frame2 = frames2[i][y_min:y_max, x_min:x_max]
                char_width, char_dist = self.__calculate_char_width(frame2)

                if char_dist > 0:
                    char_widths.append(char_width)
                    char_dists.append(char_dist)
                found_frames.append(frames2[i])

                self.logger.debug("Average character width {} ".format(char_width))
                self.logger.debug("Average character distance {} ".format(char_dist))

            batch += 1

        self.char_width = int(np.median(char_widths))
        self.char_dist = int(np.median(char_dists))
        self.subtitle_region = subtitle_region

        return found_frames
