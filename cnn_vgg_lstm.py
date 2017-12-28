import time
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model, load_model
from keras.layers import Input, Flatten, Dense, LSTM
import keras

import cv2
import os
import numpy as np
import pandas as pd

video_path = "data-samples"
m = 256
batch_size = 16
cats = {"agitated": 0,
        "feeding": 1,
        "normal": 2,
        "obstructed": 3}
model_path = "my_model_vgg16.h5"


def main():

    print("keras.__version__", keras.__version__)
    #cnn_lstm_video_classification()

    for x_train, y_train in my_generator(32):
        print(x_train.shape, y_train.shape)


def cnn_lstm_video_classification():
    model = load_vgg16_model()

    model.fit_generator(
        generator=my_generator(batch_size),
        steps_per_epoch=1,
        epochs=1,
        verbose=2
    )


    model.save(model_path)


def fit_model(model, test):

    X_train = np.zeros((0, 224, 224, 3))
    Y_train = np.zeros((0, 4), dtype=np.int)



    for dir in os.listdir(video_path):
        for file in [os.path.join(video_path, dir, f) for f in os.listdir(os.path.join(video_path, dir)) if f.endswith("normal-008.mp4")]:

            cap = cv2.VideoCapture(file)
            ret, frame = cap.read()
            width, height, channels = frame.shape
            side = min(width, height)

            x0 = int((width - side) / 2)
            x1 = int((width + side) / 2)
            y0 = int((height - side) / 2)
            y1 = int((height + side) / 2)

            i = 0
            x_train = np.zeros((m, 224, 224, 3))

            while (ret and i<m):
                ret, frame = cap.read()

                if(ret):
                    pic = cv2.resize(frame[x0:x1, y0:y1, :], (224, 224))
                    x_train[i] = pic
                    i += 1

                    #if (cv2.waitKey(1) & 0xFF == ord('q')):
                    #    break

            cap.release()
            cv2.destroyAllWindows()

            x_train = preprocess_input(x_train[:i])

            y_train = np.zeros((i,4), dtype=np.int)
            y_train[:, cats[dir]] = 1

            if(not test):
                print("fit", file)
                model.fit(
                    x=x_train,
                    y=y_train,
                    epochs=1
                )
            else:
                print("test", file)
                scores = model.evaluate(
                    x=x_train,
                    y=y_train
                )
                print(scores)


def my_generator(batch_size):
    i = 0
    x_train = np.zeros((batch_size, 224, 224, 3))
    y_train = np.zeros((batch_size, 4), dtype=np.int)

    (height, width, channels) = (1080, 1920, 3)
    side = min(height, width)

    x0 = int((width - side) / 2)
    x1 = int((width + side) / 2)
    y0 = int((height - side) / 2)
    y1 = int((height + side) / 2)

    for dir in os.listdir(video_path):
        for file in [os.path.join(video_path, dir, f) for f in os.listdir(os.path.join(video_path, dir)) if f.endswith(".mp4")]:

            print(file)
            cap = cv2.VideoCapture(file)

            while True:
                ret, frame = cap.read()
                if(ret):
                    pic = cv2.resize(frame[x0:x1, y0:y1, :], (224, 224))

                    x_train[i] = pic
                    y_train[i, cats[dir]] = 1

                    i += 1

                    if(i >= batch_size):
                        x_train = preprocess_input(x_train[:i])
                        print("before yield", x_train.shape, y_train.shape)
                        yield x_train, y_train
                        i=0
                        x_train = np.zeros((batch_size, 224, 224, 3))
                        y_train = np.zeros((batch_size, 4), dtype=np.int)
                else:
                    break
            cap.release()
            #cv2.destroyAllWindows()




def load_vgg16_model():

    if(os.path.isfile(model_path)):
        my_model = load_model(model_path)
    else:

        input = Input(shape=(224, 224, 3), name="my_input")
        base_model = VGG16(include_top=False, weights='imagenet', input_tensor=input)
        x = base_model.output

        #add FC layers
        x = Flatten(name='flatten')(x)
    #    x = LSTM(256, name='lstm1')(x)
    #    x = LSTM(256, name='lstm2')(x)
        x = Dense(4, activation='softmax', name='y_pred')(x)

        my_model = Model(inputs=base_model.input, outputs=x)
        #my_model.summary()

        for layer in base_model.layers[:15]:
            layer.trainable = False

        my_model.compile(optimizer='adam', loss='categorical_crossentropy')

        #my_model.summary()


    return my_model


if __name__ == "__main__":
    main()


