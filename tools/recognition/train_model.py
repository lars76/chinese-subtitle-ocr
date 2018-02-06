import os

import cv2
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

MODEL_FILE = "model-{val_categorical_accuracy:.2f}.h5"

TRAIN_DATA_DIR = "train/"
VALIDATION_DATA_DIR = "validation/"

EPOCHS = 500
BATCH_SIZE = 16
PATIENCE = 30


def create_model(input_shape, categories):
    # inspired by VGG-16
    model = Sequential()

    # Block 1
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", name="block1_conv1", input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", name="block1_conv2"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="block1_pool"))
    model.add(Dropout(0.25))

    # Block 2
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", name="block2_conv1"))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", name="block2_conv2"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="block2_pool"))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(categories, activation="softmax"))

    return model


def main():
    sub_folders = next(os.walk(TRAIN_DATA_DIR))[1]
    _, _, files = next(os.walk(os.path.join(TRAIN_DATA_DIR, sub_folders[0])))

    categories = len(sub_folders)
    input_shape = cv2.imread(os.path.join(TRAIN_DATA_DIR, sub_folders[0], files[0])).shape

    model = create_model(input_shape, categories)

    model.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["categorical_accuracy"])

    # export model
    yaml_string = model.to_yaml()
    with open("model.yml", "w") as file:
        file.write(yaml_string)

    train_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_DIR,
        target_size=(input_shape[0], input_shape[1]),
        batch_size=BATCH_SIZE,
        class_mode="categorical")

    validation_generator = test_datagen.flow_from_directory(
        VALIDATION_DATA_DIR,
        target_size=(input_shape[0], input_shape[1]),
        batch_size=BATCH_SIZE,
        class_mode="categorical")

    checkpoint = ModelCheckpoint(MODEL_FILE, monitor="val_categorical_accuracy", verbose=1, save_best_only=True,
                                 save_weights_only=True,
                                 mode="auto", period=1)
    stop = EarlyStopping(monitor="val_categorical_accuracy", min_delta=0, patience=PATIENCE, verbose=1, mode="auto")

    training_samples = sum(
        [len(files) for r, d, files in os.walk(os.path.join(TRAIN_DATA_DIR, os.listdir(TRAIN_DATA_DIR)[0]))])
    validation_samples = sum(
        [len(files) for r, d, files in os.walk(os.path.join(VALIDATION_DATA_DIR, os.listdir(TRAIN_DATA_DIR)[0]))])

    model.fit_generator(
        train_generator,
        steps_per_epoch=training_samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=max(validation_samples // BATCH_SIZE, 1),
        callbacks=[checkpoint, stop])

    model.summary()


if __name__ == "__main__":
    main()
