import time
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.layers import Input, Flatten, Dense, LSTM
import keras

import cv2
import os
import numpy as np
import pandas as pd

video_path = "data-samples"
m = 256
cats = {"agitated": 0,
        "feeding": 1,
        "normal": 2,
        "obstructed": 3}
model_path = "my_model.h5"

def main():

    print("keras.__version__", keras.__version__)

    cnn_lstm_video_classification()



def cnn_lstm_video_classification():
    model = load_vgg16_model()

    #fit_model(model, test=False)

    fit_model(model, test=True)

    model.save()

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


def load_vgg16_model():

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


