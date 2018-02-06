import logging

import cv2
from keras.models import model_from_yaml


class Recognition:

    def __init__(self, model_file, weights_file, dictionary):
        self.logger = logging.getLogger(__name__)

        with open(model_file, "r") as file:
            self.model = model_from_yaml(file.read())
            height = self.model.inputs[0].shape[1]
            self.img_size = (height, height)
        self.model.load_weights(weights_file)

        with open(dictionary, "r") as file:
            self.dictionary = {}
            data = file.read().split("\n")
            for index, character in enumerate(data):
                self.dictionary[index] = character

        self.logger.debug("Loaded model")

    def recognize_character(self, image, resize=True):
        if resize:
            image = cv2.resize(image, self.img_size)

        result = self.model.predict(image[None, :])
        index = result.argmax(axis=-1)[0]
        ret = (index, self.dictionary[index], result[0][index])

        logging.debug("Found {} (probability {} %)".format(ret[1], ret[2]))

        return ret
